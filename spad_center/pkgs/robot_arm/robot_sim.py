import pybullet as p
import pybullet_data
import time
import math
import urx 
import numpy as np
import contextlib, sys, os
from contextlib import contextmanager

@contextmanager
def suppress_bullet_warnings():
    devnull = open(os.devnull, 'w')
    orig_stdout, orig_stderr = os.dup(1), os.dup(2)
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(orig_stdout, 1)
        os.dup2(orig_stderr, 2)
        devnull.close()

class SnakeGridPlanner:
    def __init__(self, ul_joints_deg, ur_joints_deg, ll_joints_deg, y_steps=10, z_steps=10, urdf_path=None):
        self.ul_joints_deg = ul_joints_deg
        self.ur_joints_deg = ur_joints_deg
        self.ll_joints_deg = ll_joints_deg
        self.y_steps = y_steps
        self.z_steps = z_steps
        self.urdf_path = urdf_path
        self.grid_points = None
        self.counter = 0
        self.robot_id = None
        self.end_effector_index = 6
        self.robot = None
        self._connect_real_robot()

    def _connect_real_robot(self):
        try:
            self.robot = urx.Robot("192.168.1.52")
            print("Connected to real robot at 192.168.1.52")
        except Exception as e:
            self.robot = None
            print(f"Could not connect to real robot: {e}")

    def send_to_real_robot(self, joint_angles_deg):
        if self.robot is None:
            print("No active robot connection. Skipping real robot move.")
            return
        try:
            joint_angles_rad = [math.radians(a) for a in joint_angles_deg]
            self.robot.movej(joint_angles_rad, acc=0.1, vel=0.1)
        except Exception as e:
            print(f"Failed to move real robot: {e}")

    def check_shoulder_limits(self, min_angle=-140):
        if self.grid_points is None:
            print("Grid points not generated yet.")
            return
        violations = []
        for idx, target_pos in enumerate(self.grid_points):
            ik_solution = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, target_pos)
            angles_deg = [math.degrees(ik_solution[i]) for i in range(6)]
            if angles_deg[1] < min_angle:
                violations.append((idx, angles_deg[1]))
        if violations:
            print(f"\033[1;31mShoulder joint below {min_angle}° at points: {violations}\033[0m")
            sys.exit(0)
        else:
            print(f"\033[1;32mAll shoulder angles above {min_angle}° ✅\033[0m")

    def _apply_joint_angles(self, joint_angles_deg):
        joint_angles_rad = [math.radians(a) for a in joint_angles_deg]
        revolute_set = 0
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] == p.JOINT_REVOLUTE and revolute_set < len(joint_angles_deg):
                p.resetJointState(self.robot_id, j, joint_angles_rad[revolute_set])
                revolute_set += 1

    def _get_end_effector_pos(self):
        return p.getLinkState(self.robot_id, self.end_effector_index)[4]

    def _generate_snake_grid(self, LL, UR, UL=None):
        x = LL[0]
        y_vals = np.linspace(LL[1], UR[1], self.y_steps)
        
        if UL is not None:
            z_vals = np.linspace(LL[2], UL[2], self.z_steps)
        else:
            z_vals = np.linspace(LL[2], UR[2], self.z_steps)
        
        grid_points = []
        for i, z in enumerate(z_vals):
            row = y_vals if i % 2 == 0 else y_vals[::-1]
            grid_points.extend([(x, y, z) for y in row])
        
        self.grid_points = np.array(grid_points)


    def initialize_simulation(self):
        with suppress_bullet_warnings():
            p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
            self._apply_joint_angles(self.ur_joints_deg)
            p.stepSimulation()
            ur_pos = self._get_end_effector_pos()
            self._apply_joint_angles(self.ll_joints_deg)
            p.stepSimulation()
            ll_pos = self._get_end_effector_pos()
            self._apply_joint_angles(self.ul_joints_deg)
            p.stepSimulation()
            ul_pos = self._get_end_effector_pos()
            self._generate_snake_grid(ll_pos, ur_pos, ul_pos)
            
    def show_simulation(self):
        p.disconnect()
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
        revolute_joints = [j for j in range(p.getNumJoints(self.robot_id))
                           if p.getJointInfo(self.robot_id, j)[2] == p.JOINT_REVOLUTE]
        for step, target_pos in enumerate(self.grid_points, 1):
            ik_solution = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, target_pos)
            for idx, joint_index in enumerate(revolute_joints):
                p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL,
                                        targetPosition=ik_solution[idx], force=500)
            for _ in range(120):
                p.stepSimulation()
                time.sleep(1./240.)

    def get_next_angles(self):
        if self.counter >= len(self.grid_points):
            return None
        target_pos = self.grid_points[self.counter]
        ik_solution = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, target_pos)
        angles_deg = [math.degrees(ik_solution[i]) for i in range(6)]
        self.counter += 1
        return angles_deg


if __name__ == "__main__":
    LL_JOINTS_DEG = [90, -110.82, 151.63, -82.93, -180, 225.22]  # Lower Left pose
    UR_JOINTS_DEG = [90, -45.64, 31.43, -64.68, -180, 225.22]  
    UL_JOINTS_DEG = [90, -129.53, 123.36, -87.48, -180, 225.22]
    urdf_path = "./robot_arm/ur10_robot.urdf"

    planner = SnakeGridPlanner(UL_JOINTS_DEG, UR_JOINTS_DEG, LL_JOINTS_DEG, y_steps=8, z_steps=4, urdf_path=urdf_path)
    planner.initialize_simulation()
    planner.check_shoulder_limits()
    # sys.exit(0)
 
    print("Iterating through joint angles:")
    for _ in range(len(planner.grid_points)):
        angles = planner.get_next_angles()
        planner.send_to_real_robot(angles)
        print(np.round(angles, 4))

    # planner.show_simulation()