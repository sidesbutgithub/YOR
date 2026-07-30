import time
from pathlib import Path
from typing import Optional
import numpy as np
import mink

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from robot.arm.gripper import Gripper
from robot.arm.ik_solver import SingleArmIK

# Dynamixel gripper caliberated in dxl.py
# DXL_ID_RIGHT = 2, DXL_ID_LEFT = 3
BAUDRATE = 1000000 
DXL_ID_RIGHT = 2
DXL_ID_LEFT = 3

class ArmNode:
    def __init__(
        self,
        can_port: str,
        mjcf_path: str,
        solver_dt: float = 0.01,
        is_left_arm: bool = True,
        dynamixel_gripper: bool = False,
        default_kp: Optional[float | list[float]] = 15.0,
        default_kd: Optional[float | list[float]] = 0.8,
        gravity_comp_scale: float = 1.0,
        firmware_version=None,
    ):
        _ROOT = Path(__file__).parent.parent.parent
        self.can_port = can_port
        self.is_left_arm = is_left_arm
        if is_left_arm:
            self.urdf_path = (_ROOT / "nerolib/urdf/nero_cone-e_left_fixed.urdf").as_posix()
        else:
            self.urdf_path = (_ROOT / "nerolib/urdf/right_arm_final.urdf").as_posix()
        self.solver_dt = solver_dt

        # Initialize nerolib NeroController
        self.control_mode_set = False
        try:
            print(f"[ArmNode] Initializing {can_port} with nerolib...")

            """
            self.config = ControllerConfig()
            self.config.interface_name = can_port
            self.config.urdf_path = self.urdf_path
            """

            # Defines home position (originally in ControllerConfig)
            # UPDATED: Using a safe intermediate home based on current readouts to avoid J1 limits/issues
            self.home_position = (
                [0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0] 
                # [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Modified J1 from 1.38 to -1.38 to match current state sign
                if is_left_arm
                else [0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0]
            )
            
            #self.config.home_position = self.home_position
            
            self.joint_vel_max = [3.0] * 7
            self.joint_acc_max = [15.0] * 7
            
            self.torque_ranges = [24.0, 24.0, 16.0, 16.0, 8.0, 8.0, 8.0]

            if default_kp is not None:
                if isinstance(default_kp, (int, float)):
                    self.default_kp = [float(default_kp)] * 7
                else:
                    self.default_kp = [float(x) for x in default_kp]
            
            if default_kd is not None:
                if isinstance(default_kd, (int, float)):
                    self.default_kd = [float(default_kd)] * 7
                else:
                    self.default_kd = [float(x) for x in default_kd]

            self.gravity_compensation = False
            self.gravity_comp_scale = gravity_comp_scale

            #if firmware_version is not None:
            #    self.firmware_version = firmware_version

            self.config = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT, channel=can_port)

            #self.nero = NeroController(self.config)
            
            self.nero = AgxArmFactory.create_arm(self.config)
            self.nero.connect()

            err = self.nero.get_comm_error()
            if err is not None:
                raise Exception("Nero Arm Communication Error:", err)

            max_retries = 100
            curr_retries = 0
            while not self.nero.enable() and curr_retries < max_retries:
                curr_retries+=1
                time.sleep(0.01)

            if not curr_retries < max_retries:
                print(f"[ArmNode] Failed to enable pyAgxArm arm on {can_port}")
                self.nero = None
            else:
                print(f"[ArmNode] {can_port} initialized and started.")
                self.control_mode_set = True
            
        except Exception as e:
            print(f"[ArmNode] Failed to initialize pyArxArm on {can_port}: {e}")
            self.nero = None

        #self.set_vel_and_acc_limits(vel_limits=self.joint_vel_max, acc_limits=self.joint_acc_max)

        self.target: Optional[mink.SE3] = None
        self.gripper_target: Optional[float] = None
        self.q_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if is_left_arm:
            joint_names = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ]
            ee_frame = "left_arm_ee"
        else:
            joint_names = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ]
            ee_frame = "right_arm_ee"

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )

        self.dynamixel_gripper = dynamixel_gripper
        self.gripper = None

        if self.dynamixel_gripper:
            # If your Gripper class uses /dev/ttyUSB0 internally, this will fail when unplugged.
            # Auto-disable instead of crashing.
            try:
                dxl_id = DXL_ID_LEFT if is_left_arm else DXL_ID_RIGHT
                self.gripper = Gripper(baudrate=BAUDRATE, dxl_id=dxl_id)

                open_gripper_value, close_gripper_value = self.gripper.dxl.calibrate_motor()
                self.open_gripper_value = open_gripper_value
                self.close_gripper_value = close_gripper_value
                self.gripper_range = open_gripper_value - close_gripper_value

                print("[ArmNode] Dynamixel gripper enabled")

            except (FileNotFoundError, OSError) as e:
                print(f"[ArmNode] Dynamixel gripper not found ({e}). Continuing without gripper.")
                self.dynamixel_gripper = False
                self.gripper = None

    def init(self):
        if self.nero is None:
            print("[ArmNode] Warning: Nero not initialized, init() skipping hardware calls.")
            return

        # Move to home position
        print(f"[ArmNode] Moving to home position...")
        self.reset_to_home()
        
        # Sync IK solver with current state
        q = self.get_joint_positions()
        
        print(f"q_reached: {np.round(q, 4)}")
        
        self.ik_solver.init(q)
        
        self.target = self.ik_solver.forward_kinematics()
        
    def reset_to_home(self):
        self.set_joint_target(joint_target=np.array(self.home_position, dtype=float))

    def home(self, gripper_target: float = 1.0):
        if self.nero:
            try:
                self.reset_to_home()
            except Exception as e:
                print(f"[ArmNode] Home failed: {e}")
        
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))
        time.sleep(2.0)
        self.update_joint_positions()

    def tuck_arms(self):
        self.set_joint_target(np.zeros(7), gripper_target=1.00, preview_time=2.0)

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.nero:
            try:
                # nerolib expects lists or std::vectors, typically python lists/arrays work with pybind11
                # The API signature: set_target(new_target_pos, new_target_gripper_pos, minimum_duration, new_target_vel, new_target_acc)
                # We map joint_target to new_target_pos.
                
                target_pos = (joint_target + self.q_offset).tolist()
                
                # Nerolib expects normalized gripper position (0 for close, 1 for fully open)
                target_gripper = 1.0
                if gripper_target is not None:
                    target_gripper = float(gripper_target)
                
                self.nero.move_j(target_pos)

                """
                self.nero.set_target(
                    new_target_pos=target_pos,
                    new_target_gripper_pos=target_gripper if not self.dynamixel_gripper else 0.0, # Don't conflict if dyn gripper used
                    minimum_duration=preview_time
                )
                """
                
            except Exception as e:
                 print(f"Set target failed: {e}")

        if gripper_target is not None and self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))

    def open_gripper(self):
        if not self.dynamixel_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=1.0)  
            time.sleep(0.5)
        else:
            self.gripper.move_to_pos(self.open_gripper_value)

    def close_gripper(self):
        if not self.dynamixel_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=0.0)
            time.sleep(0.5)
        else:
            self.gripper.move_to_pos(self.close_gripper_value)

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None, preview_time: float = 0.0):
        self.target = ee_target
        qd, _ = self.ik_solver.solve_ik(self.target)
        self.set_joint_target(qd, gripper_target, preview_time)
        if gripper_target is not None and self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * (self.gripper_range) + self.close_gripper_value))

    def torque_control(self, joint_index, coefficient):
        self.nero.move_mit(kp=0, kd=0, joint_index=joint_index, t_ff=coefficient*self.torque_ranges[joint_index-1])

    #def set_gain(self, kp, kd):
        """
        Set Kp and Kd gains for the controller.
        kp, kd can be float (applied to all joints) or 7-element lists/arrays.
        """
    #    if self.nero:
    #        gain = Gain()
    #        if isinstance(kp, (int, float)):
    #            gain.kp = [float(kp)] * 7
    #        else:
    #            gain.kp = [float(x) for x in kp]
    #        
    #        if isinstance(kd, (int, float)):
    #            gain.kd = [float(kd)] * 7
    #        else:
    #            gain.kd = [float(x) for x in kd]
    #        
    #        self.nero.set_gain(gain)

    def set_gravity_comp(self, enable: bool):
        if self.nero:
            self.nero.enable_gravity_compensation(enable)

    def set_gravity_comp_scale(self, scale: float):
        if self.nero:
            self.nero.set_gravity_comp_scale(scale)

    #def set_mode(self, control_mode: ControlMode, move_mode: MoveMode):
    #    if self.nero:
    #        self.nero.set_mode(control_mode, move_mode)

    def set_compliant_mode(self, kp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], kd=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]):
        """preset for manual guidance. Default: Stiffness=0, Damping=0.5."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def set_spring_mode(self, kp=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], kd=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]):
        """preset for springy behavior. Default: Stiffness=2, Damping=0.3."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def set_firm_mode(self, kp=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], kd=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]):
        """preset for accurate tracking. Default: Stiffness=15, Damping=0.8."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0 )

    def set_admittance_mode(self, kp=[25.0, 25.0, 20.0, 20.0, 15.0, 15.0, 15.0], kd=[1.2, 1.2, 1.0, 1.0, 0.8, 0.8, 0.8]):
        """
        High stiffness and damping, optimized for a high-level admittance loop.
        """
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def sync_target(self):
        """Update the controller's target to match the current arm position."""
        if self.nero:
            q = self.get_joint_positions()
            self.nero.set_target(
                new_target_pos=(q + self.q_offset).tolist(),
                new_target_gripper_pos=self.get_gripper_pose(),
                minimum_duration=0.0
            )
            # Give the C++ control loop (500Hz) a few cycles to ingest this 
            # and 'snap' its internal trajectory state before we change gains.
            time.sleep(0.02)

    def get_joint_positions(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:
            ja = self.nero.get_joint_angles()
            q = np.array(ja.msg)
            return q - self.q_offset
        except Exception:
            return np.zeros(7)

    def get_joint_velocities(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:
            vel_arr = [self.nero.get_motor_states(joint_index = i).msg.velocity for i in range(1, self.nero.joint_nums+1)]
            return np.array(vel_arr)
        except Exception:
            return np.zeros(7)

    def get_joint_torques(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:

            torque_arr = [self.nero.get_motor_states(joint_index = i).msg.torque for i in range(1, self.nero.joint_nums+1)]
            return np.array(torque_arr)
        except Exception:
            return np.zeros(7)

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()

    def get_gripper_pose(self):
        if not self.dynamixel_gripper:
            # not implemented for pyAgxArm
            if self.nero:
                return self.get_ee_pose()
            return 0.0 
        else:
            unnormalized_pos = self.gripper.dxl.get_present_position()
            return float(unnormalized_pos - self.close_gripper_value) / float(
                self.gripper_range
            ) 

    def update_joint_positions(self):
        q = self.get_joint_positions()
        self.ik_solver.update_configuration(q)

    def set_q_offset(self, q_offset: np.ndarray):
        self.q_offset = q_offset

    def stop(self):
        print("called stop")
        if self.nero:
             self.nero.disable()

    # helper to set limits for pyAgxArm api
    def set_vel_and_acc_limits(self, vel_limits, acc_limits):
        for joint_index in range(1, self.nero.joint_nums + 1):
            print("attempting vel limit set:", vel_limits[joint_index-1])
            self.nero.set_joint_angle_vel_limits(joint_index=joint_index, max_joint_spd=vel_limits[joint_index-1])
            print("attempting acc limit set:", acc_limits[joint_index-1])
            self.nero.set_joint_acc_limits(joint_index=joint_index, max_joint_acc=acc_limits[joint_index-1])

if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    left_arm = ArmNode(
        can_port="can_left",
        mjcf_path=(_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
        is_left_arm=True,
    )
    # right_arm = ArmNode(
    #     can_port="can_right",
    #     mjcf_path=(_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
    #     is_left_arm=False,
    # )
    input("Press enter to init left arm")
    left_arm.init()
    # input("Press enter to init right arm")
    # right_arm.init()
    input("Press enter to close gripper")
    left_arm.close_gripper()
    # right_arm.close_gripper()
    input("Press enter to open gripper")
    left_arm.open_gripper()
    # right_arm.open_gripper()
    input("Press enter to exit")
    left_arm.stop()
    # right_arm.stop()
    exit()
