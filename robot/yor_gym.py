import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from gymnasium import error, logger, spaces 
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import math
from typing import Any, Optional, Tuple, Union

import numpy as np

import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
import threading
from typing import Optional
from pathlib import Path
import atexit

from loop_rate_limiters import RateLimiter

from robot.arm.ik_solver import SingleArmIK
from commlink import RPCServer

_HERE = Path(__file__).parent

def wrap_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def diff_angle(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
    return ((a - b) + math.pi) % (2 * math.pi) - math.pi


def frac_to_rad(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return ((np.array(f) + 0.5) % 1.0 - 0.5) * TWO_PI


def rad_to_frac(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.array(rad) / TWO_PI) % 1.0

NUM_SWERVES = 4
LENGTH = 0.1225  # m
WIDTH = 0.170  # m
TIRE_RADIUS = 0.0381  # m

MODULE_ORDER = ("FL", "FR", "RR", "RL")

DRIVE_NAMES = ("drive_front_left_ctrl", "drive_front_right_ctrl", "drive_back_right_ctrl", "drive_back_left_ctrl")  # [FL, FR, RR, RL]
ROT_NAMES = ("front_left_steer_ctrl", "front_right_steer_ctrl", "back_right_steer_ctrl", "back_left_steer_ctrl")  # [FL, FR, RR, RL]

ROTATION_OFFSETS = np.array([0.75, 0.00, 0.25, 0.50], dtype=float)

ROT_DIAG_SWAP_PERM = np.array([1, 0, 3, 2], dtype=int)
TRANS_OPPOSITE_MASK = np.array([False, False, False, False], dtype=bool)

TWO_PI = 2.0 * math.pi

USE_FEEDBACK_FOR_STEER = False
DRIVE_VEL_SCALE = 2.0

TIRE_CIRCUMFERENCE = TWO_PI*TIRE_RADIUS

def mps_to_rad_ps(mps):
    return mps/TIRE_CIRCUMFERENCE*TWO_PI

LIFT_MAX = 0.416
LIFT_MIN = 0

class YORMujocoRaw:
    def __init__(self,xml_path,env_limit=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def reset(self):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0
    
        mujoco.mj_forward(self.model,self.data)
    
    def get_obs(self,noisy=False,use_obs=False,noise_scale=0.01):
        return np.concatenate([self.data.qpos,self.data.qvel])

class YORMujocoBase():
    def __init__(self, raw):
        self.raw = raw
        self.model = self.raw.model
        self.data = self.raw.data

        self.rotation_motors = [self.model.actuator(i).id for i in ROT_NAMES]# [FL, FR, RR, RL]
        self.drive_motors = [self.model.actuator(i).id for i in DRIVE_NAMES]# [FL, FR, RR, RL]

        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)
        self.x = np.zeros(3)
        self.dx = np.zeros(3)

        self.dt = self.model.opt.timestep
        self.policy_control_period_ns = int(1e9 / (1/self.dt))
        self.last_command_time_ns = 0.0

        self.base_target = np.zeros(3)

        # --- S-curve profiling state (kept; now optional per-command) ---
        self._smooth_active = False  # whether to apply smoothing for the *current* command
        self._v_prof = np.zeros(3, dtype=float)
        self._seg_v0 = np.zeros(3, dtype=float)
        self._seg_v1 = np.zeros(3, dtype=float)
        self._seg_t = 0.0
        self._seg_T = 0.0

        self._a_max = np.array([1.9, 1.9, 6.5], dtype=float)
        self._T_min = 0.01
        self._retarget_eps = 1e-3

    def control_iteration(self):
        disable_motors = False
        #print(time.perf_counter_ns() - self.last_command_time_ns, self.policy_control_period_ns)
        if (time.perf_counter_ns() - self.last_command_time_ns) > 10 * self.policy_control_period_ns:

            disable_motors = True

        self._smooth_active = True # always true in physical

        self._update_state()

        if disable_motors:
            for i, dm in enumerate(self.drive_motors):
                self.data.ctrl[dm] = float(0.0)
        else:
            v_cmd = self.base_target
            if self._smooth_active:
                if np.linalg.norm(v_cmd - self._seg_v1) > self._retarget_eps:
                    self._start_scurve_segment(v_cmd)
                v_used = self._update_scurve(self.dt)
            else:
                # Keep profiling state consistent so enabling smoothing later doesn't jump from stale state
                self._v_prof = v_cmd.copy()
                self._seg_v0 = v_cmd.copy()
                self._seg_v1 = v_cmd.copy()
                self._seg_t = 0.0
                self._seg_T = 0.0
                v_used = v_cmd

            wheel_speeds, wheel_angles = self._vehicle_velocity_to_angle_and_speed(
                v_used, cos_error_scaling=True
            )
            target_fracs = rad_to_frac(wheel_angles)
            for i, rm in enumerate(self.rotation_motors):
                self.data.ctrl[rm] = float(wheel_angles[i])

            for i, dm in enumerate(self.drive_motors):
                self.data.ctrl[dm] = float(mps_to_rad_ps(wheel_speeds[i]))
            #print(wheel_speeds, target_fracs)

    # -------------- helpers --------------
    def _update_state(self) -> None:

        for i, rm in enumerate(self.rotation_motors):
            self.steer_pos[i] = self.data.ctrl[rm]

        for i, dm in enumerate(self.drive_motors):
            self.drive_vel[i] = self.data.ctrl[dm]

    def _angle_and_speed_to_vehicle_velocity(
        self, wheel_speeds: np.ndarray, wheel_angles: np.ndarray
    ) -> np.ndarray:
        vx, vy = wheel_speeds * np.cos(wheel_angles), wheel_speeds * np.sin(wheel_angles)
        return np.linalg.lstsq(self.C, np.concatenate((vx, vy)), rcond=None)[0]

    def _start_scurve_segment(self, v_target: np.ndarray):
        v_target = np.asarray(v_target, dtype=float)

        if getattr(self, "_seg_T", 0.0) > 0 and np.allclose(v_target, self._seg_v1, atol=1e-3):
            return

        dv = v_target - self._v_prof
        abs_dv = np.abs(dv)

        if np.all(abs_dv < 1e-3):
            return

        T_needed = np.max((abs_dv * np.pi) / (2.0 * np.maximum(self._a_max, 1e-6)))
        T = max(self._T_min, float(T_needed))

        self._seg_v0 = self._v_prof.copy()
        self._seg_v1 = v_target.copy()
        self._seg_t = 0.0
        self._seg_T = T

    def _update_scurve(self, dt: float) -> np.ndarray:
        if self._seg_T <= 1e-9:
            return self._v_prof

        self._seg_t = min(self._seg_t + dt, self._seg_T)
        tau = self._seg_t / self._seg_T
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        self._v_prof = self._seg_v0 + (self._seg_v1 - self._seg_v0) * s
        return self._v_prof

    def _vehicle_velocity_to_angle_and_speed(
        self, u_3dof: np.ndarray, cos_error_scaling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, omega = float(u_3dof[0]), float(u_3dof[1]), float(u_3dof[2])

        vx_t = np.array([vx, vx, vx, vx], dtype=float)
        vy_t = np.array([vy, vy, vy, vy], dtype=float)
        sign = np.where(TRANS_OPPOSITE_MASK, -1.0, 1.0)
        vx_t *= sign
        vy_t *= sign

        vx_r = np.array(
            [+WIDTH * omega, -WIDTH * omega, -WIDTH * omega, +WIDTH * omega], dtype=float
        )
        vy_r = np.array(
            [+LENGTH * omega, +LENGTH * omega, -LENGTH * omega, -LENGTH * omega],
            dtype=float,
        )
        vx_r = vx_r[ROT_DIAG_SWAP_PERM]
        vy_r = vy_r[ROT_DIAG_SWAP_PERM]

        vx_w = vx_t + vx_r
        vy_w = vy_t + vy_r

        wheel_speeds = np.hypot(vx_w, vy_w)
        wheel_angles = np.arctan2(vy_w, vx_w)

        error = diff_angle(wheel_angles, self.steer_pos)
        wheel_angles = np.where(
            np.abs(error) > np.pi / 2, diff_angle(wheel_angles, np.pi), wheel_angles
        )
        wheel_speeds = np.where(np.abs(error) > np.pi / 2, -wheel_speeds, wheel_speeds)

        if cos_error_scaling:
            wheel_speeds *= np.cos(diff_angle(wheel_angles, self.steer_pos))

        return wheel_speeds, wheel_angles

    def _map_steer_angles(self, wheel_angles: np.ndarray) -> np.ndarray:
        ang = wheel_angles.copy()
        ang[TRANS_OPPOSITE_MASK] = ang[TRANS_OPPOSITE_MASK] + math.pi
        ang = ang[ROT_DIAG_SWAP_PERM]
        return wrap_pi(ang)

class YORMujocoController():
    def __init__(
        self,
        no_arms: bool = False,  # default to True since Base is independent now
        mjcf_path: str = "", solver_dt: float = 0.01, env_limit: int  = 10,
        max_vel=np.array((1.0, 1.0, 1.57)),
        max_accel=np.array((1.0, 1.0, 1.57)),
        render: bool = True
    ):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # launch mujoco
        self.raw = YORMujocoRaw(self.mjcf_path, self.solver_dt)
        self.model = self.raw.model
        self.data = self.raw.data
        self.render = render
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.left_q_desired: Optional[np.ndarray] = None
        self.left_q_desired_lock = threading.Lock()
        self.left_ik_solver = SingleArmIK(
            (_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
            solver_dt=self.solver_dt,
            joint_names=[
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ],
            ee_frame="left_arm_ee",
        )

        self.right_q_desired: Optional[np.ndarray] = None
        self.right_q_desired_lock = threading.Lock()
        self.right_ik_solver = SingleArmIK(
            (_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
            solver_dt=self.solver_dt,
            joint_names=[
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ],
            ee_frame="right_arm_ee",
        )

        # initialize base
        self.base = YORMujocoBase(self.raw)

        # lift target height
        self.lift_delta = 0
        self.lift_target = 0
        self.lift_id = self.model.actuator("lift").id

        self.control_loop_thread: threading.Thread | None = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self.left_ik_solver.update_configuration(self.data.qpos.copy())
        qd, is_solved = self.left_ik_solver.solve_ik(ee_target)
        print(f"desired q: {np.round(qd, 4)} | is_solved: {is_solved}")
        with self.left_q_desired_lock:
            self.left_q_desired = qd

    def set_left_joint_target(self, joint_target: np.ndarray):
        with self.left_q_desired_lock:
            self.left_q_desired = joint_target

    def get_left_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self.left_ik_solver.dof_ids]

    def get_left_ee_pose(self) -> mink.SE3:
        q = self.data.qpos.copy()
        self.left_ik_solver.update_configuration(q)
        return self.left_ik_solver.forward_kinematics()

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self.right_ik_solver.update_configuration(self.data.qpos.copy())
        qd, is_solved = self.right_ik_solver.solve_ik(ee_target)
        print(f"desired q: {np.round(qd, 4)} | is_solved: {is_solved}")
        with self.right_q_desired_lock:
            self.right_q_desired = qd

    def set_right_joint_target(self, joint_target: np.ndarray):
        with self.right_q_desired_lock:
            self.right_q_desired = joint_target

    def get_right_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self.right_ik_solver.dof_ids]

    def get_right_ee_pose(self) -> mink.SE3:
        q = self.data.qpos.copy()
        self.right_ik_solver.update_configuration(q)
        return self.right_ik_solver.forward_kinematics()

    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new instance of ArmMujoco first")
            return
        # self.init()
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("Control thread not running")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None
        self.viewer.close()

    def init(self):
        self.start_control()
        time.sleep(0.1)
        # home
        q = self.data.qpos.copy()
        self.left_ik_solver.init(q)
        self.right_ik_solver.init(q)
        with self.left_q_desired_lock:
            self.left_q_desired = q[self.left_ik_solver.dof_ids]
        with self.right_q_desired_lock:
            self.right_q_desired = q[self.right_ik_solver.dof_ids]

    def home_left_arm(self):
        with self.left_q_desired_lock:
            self.left_q_desired = self.left_ik_solver.get_home_q()

    def home_right_arm(self):
        with self.right_q_desired_lock:
            self.right_q_desired = self.right_ik_solver.get_home_q()

    def set_base_velocity(self, velocity: np.ndarray):
        self.base.last_command_time_ns = time.perf_counter_ns()
        self.base.base_target = velocity

    def lift_up(self):
        self.lift_delta = 0.001

    def lift_down(self):
        self.lift_delta = -0.001
    
    def lift_stop(self):
        self.lift_delta = 0

    def control_loop(self):
        rate_limiter = RateLimiter(200)
        while self.control_loop_running:
            # move arms to desired
            if self.left_q_desired is not None:
                with self.left_q_desired_lock:
                    self.data.ctrl[self.left_ik_solver.actuator_ids] = self.left_q_desired
            if self.right_q_desired is not None:
                with self.right_q_desired_lock:
                    self.data.ctrl[self.right_ik_solver.actuator_ids] = self.right_q_desired

            # move base to desired
            self.base.control_iteration()

            # move lift to desired
            self.lift_target = max(min(self.lift_target+self.lift_delta, 1.0), 0.0)
            self.data.ctrl[self.lift_id] = self.lift_target * 0.416

            # step mujoco
            mujoco.mj_step(self.model, self.data)
            if self.render:
                self.viewer.sync()
            rate_limiter.sleep()



class YORGymEnv(gym.Env):
    env_limit = 10
    distance_threshold = 0.5
    def __init__(self,max_steps=1000,
                use_orientation=False,noise_scale=0.01,
                return_full_trajectory=False, max_speed=1.0, max_steering_angle=1.0,prop_steps=100):
        self.max_steps = max_steps
        self.yor = BaseMujoco((_HERE / "yor-description" / "scene.mjcf").as_posix(),self.env_limit)
        self.yor.reset()
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))

        self.obs_dims = 3
        self.goal_dims = 3 if use_orientation else 2

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,))
        })

        self.use_orientation = use_orientation
        self.return_full_trajectory = return_full_trajectory
        
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle

        self.prop_steps = prop_steps

        self.render_mode = 'human'
        self.mujoco_renderer = MujocoRenderer(
            self.yor.model,
            self.yor.data,
            camera_name="track"
        )

    def reset(self,goal=None):
        self.yor.reset()
        self.steps = 0
        if goal is  None:
            self.goal = np.random.uniform(-self.env_limit,self.env_limit,size=(self.goal_dims,))
            if self.use_orientation:
                self.goal[2] = np.random.uniform(-np.pi,np.pi)
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.yor.get_obs()
        
        if self.use_orientation:
            achieved_goal = np.array([obs[0],obs[1],quat2euler(obs[3:7])[2]])
        else:
            achieved_goal = np.array([obs[0],obs[1]])  
        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        } 

    def _terminal(self,s,g):
        return goal_distance(s,g) < self.distance_threshold

    def compute_reward(self,ag,dg,info):
        return -(goal_distance(ag,dg) >= self.distance_threshold).astype(np.float32)

    def step(self,action):
        self.steps += 1
        
        applied_action = np.zeros_like(action)
        applied_action[0] = action[0]*self.max_speed
        applied_action[1] = action[1]*self.max_speed
        applied_action[2] = action[2]*self.max_steering_angle
        self.yor.apply_action(applied_action)
        
        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.yor.model.nv): self.yor.data.qacc_warmstart[i] = 0 
            mujoco.mj_step(self.yor.model,self.yor.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"],obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self._terminal(obs["achieved_goal"],obs["desired_goal"]) or self.steps >= self.max_steps
        reward = self.compute_reward(obs["achieved_goal"],obs["desired_goal"],{})

        self.render()
        return obs,reward,done,info
    
    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def quat2euler(q_mj):
    q_scipy = np.array([q_mj[1],q_mj[2],q_mj[3],q_mj[0]])
    r = R.from_quat(q_scipy)
    return r.as_euler('xyz',degrees=False)


if __name__ == "__main__":     
    _HERE = Path(__file__).parent
    yor_mujoco = YORMujocoController(mjcf_path=(_HERE / "yor-description" / "scene.mjcf").as_posix())
    rpc_server = RPCServer(yor_mujoco, 8081, threaded=False)
    print("Listening on port 8081")
    atexit.register(rpc_server.stop)
    rpc_server.start()

