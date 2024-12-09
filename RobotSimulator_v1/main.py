import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from abc import ABC, abstractmethod

class Block(ABC):
    @abstractmethod
    def load_block(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class Cube(Block):
    def __init__(self, initial_position=None, initial_orientation=None):
        if initial_position is None:
            initial_position = [0, 0, 0.025]
        if initial_orientation is None:
            initial_orientation = [0, 0, 0, 1]
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.block_id = self.load_block()

    def load_block(self):
        block_id = p.loadURDF("cube_small.urdf", self.initial_position, self.initial_orientation)
        p.changeDynamics(block_id, -1, mass=0.1, lateralFriction=1.0)
        return block_id

    def reset(self):
        p.resetBasePositionAndOrientation(self.block_id, self.initial_position, self.initial_orientation)

class ThreeFingerGripper:
    def __init__(self, grip_force=100):
        self.grip_force = grip_force
        self.gripper_id = self.load_gripper()
        self.num_joints = p.getNumJoints(self.gripper_id)
        print(f"Number of joints: {self.num_joints}")
        
        # Initialize all joints to closed position
        self.default_joint_positions = [0.05] * self.num_joints
        self.open = False
        
        # Create permanent base constraint
        pos = [0, 0, 0.2]
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        self.base_constraint = p.createConstraint(
            self.gripper_id, -1, -1, -1,
            p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], pos,
            childFrameOrientation=orn
        )
        p.changeConstraint(self.base_constraint, pos, orn, maxForce=100)
        
        # Disable collisions with plane
        self.disable_plane_collisions()
        
        # Set initial joint positions
        for joint_index in range(self.num_joints):
            p.resetJointState(self.gripper_id, joint_index, self.default_joint_positions[joint_index])
            
        self.enable_friction()

    def disable_plane_collisions(self):
        """Disable collisions between gripper and ground plane"""
        # Ground plane has ID 0
        for link in range(-1, self.num_joints):
            p.setCollisionFilterPair(0, self.gripper_id, -1, link, enableCollision=0)

    def load_gripper(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "Robots/grippers/threeFingers/sdh/sdh.urdf")
        initial_pos = [0, 0, 0.21]
        initial_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        gripper_id = p.loadURDF(path, initial_pos, initial_orn, useFixedBase=False)
        return gripper_id

    def enable_friction(self):
        for link in range(self.num_joints):
            p.changeDynamics(self.gripper_id, link, lateralFriction=5.0)

    def move_to(self, target_pos, target_orn, max_force=50):
        p.changeConstraint(
            self.base_constraint,
            target_pos,
            jointChildFrameOrientation=target_orn,
            maxForce=max_force
        )
        time.sleep(0.01)

    def lift_gripper(self, height=0.7, max_force=500):
        """Lift gripper by changing constraint position with appropriate force."""
        current_pos, current_orn = p.getBasePositionAndOrientation(self.gripper_id)
        
        # Move in steps to ensure stability
        steps = 10
        current_height = current_pos[2]
        height_increment = (height - current_height) / steps
        
        for i in range(steps):
            target_height = current_height + height_increment
            target_pos = [current_pos[0], current_pos[1], target_height]
            p.changeConstraint(
                self.base_constraint, 
                target_pos,
                current_orn,
                maxForce=max_force
            )
            for _ in range(120):  # Allow physics to settle
                p.stepSimulation()
                time.sleep(1/240)
            current_height = target_height

        print(f"Lifted gripper to height: {height}")

    def close_gripper(self):
        # Close main finger joints
        for joint_id in [1, 4, 7]:
            p.setJointMotorControl2(
                self.gripper_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=0.05,
                maxVelocity=1.0,
                force=self.grip_force
            )
        
        # Allow time for closure with physics steps
        for _ in range(480):  # 2 seconds at 240Hz
            p.stepSimulation()
            time.sleep(1/240)
        self.open = False

    def open_gripper(self):
        """Open gripper using example code logic"""
        closed = True
        iteration = 0
        while(closed and not self.open):
            joints = self.getJointPosition()
            closed = False
            for k in range(self.num_joints):
                #lower finger joints
                if k in [2, 5, 8]:
                    goal = 0.9
                    if joints[k] >= goal:    
                        p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05, 
                                            maxVelocity=2, force=5)   
                        closed = True
                #Upper finger joints             
                elif k in [3, 6, 9]:
                    goal = 0.9
                    if joints[k] <= goal:
                        p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2, force=5)
                        closed = True
                #Base finger joints
                elif k in [1, 4, 7]:
                    pos = 0.9
                    if joints[k] <= pos:
                        p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2, force=5)
                        closed = True
            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
        self.open = True

    def getJointPosition(self):
        joints = []
        for i in range(self.num_joints):
            pos = p.getJointState(self.gripper_id, i)[0]
            joints.append(pos)
        return joints

    def preshape_gripper(self):
        # Set preshape position for secondary joints
        for joint_id in [2, 5, 8]:
            p.setJointMotorControl2(
                self.gripper_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=0.4,
                maxVelocity=2.0,
                force=self.grip_force
            )
        
        # Allow time for preshaping
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1/240)

    def reset(self, position, orientation):
        # Reset gripper position and orientation using constraint
        p.changeConstraint(
            self.base_constraint,
            position,
            jointChildFrameOrientation=orientation,
            maxForce=100
        )
        
        # Reset joint positions
        for joint_index, position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, position)
        
        # Allow time for reset
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240)

class GraspSimulator:
    def __init__(self, num_trials=10):
        self.num_trials = num_trials
        self.setup_simulation()
        self.gripper = ThreeFingerGripper(grip_force=100)
        self.block = Cube()
        self.results = []

    def setup_simulation(self):
        # Initialize GUI physics client
        physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1/240.0)
        # Load ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

    def attempt_grasp(self, position, orientation):
        print("\nStarting grasp attempt...")
        
        self.gripper.reset(position, orientation)
        
        # Open gripper before spawning block
        print("Opening gripper...")
        self.gripper.open_gripper()
        
        # Debug print joint positions
        joints = self.gripper.getJointPosition()
        print("Joint positions after opening:", joints)
        
        self.block.reset()
        time.sleep(0.5)
        
        # Increase grip force for more reliable closing
        self.gripper.grip_force = 500  # Increase grip force
        self.gripper.close_gripper()
        
        # Debug print joint positions
        joints = self.gripper.getJointPosition()
        print("Joint positions after closing:", joints)
        
        # Increase maxForce for lifting
        self.gripper.lift_gripper(height=0.4, max_force=1000)
        time.sleep(2.0)
        
        # Check grasp success
        contact_points = p.getContactPoints(self.gripper.gripper_id, self.block.block_id)
        block_pos = p.getBasePositionAndOrientation(self.block.block_id)[0]
        success = len(contact_points) > 0 and block_pos[2] > 0.1
        
        print(f"Contact points: {len(contact_points)}")
        print(f"Block height: {block_pos[2]}")
        print(f"Grasp {'successful' if success else 'failed'}")
        
        return success

    def reset_for_next_trial(self):
        # Reset block first
        self.block.reset()
        
        # Generate noisy pose
        base_pos = [0, 0, 0.2]  # Lower initial position
        base_euler = [np.pi, 0, 0]
        
        # Add noise
        pos_noise = np.random.normal(0, 0.02, 3)
        rot_noise = [
            np.random.normal(0, 0.1),
            np.random.normal(0, 0.1),
            np.random.normal(0, np.pi)
        ]
        
        # Apply noise
        noisy_pos = [base_pos[i] + pos_noise[i] for i in range(3)]
        noisy_euler = [base_euler[i] + rot_noise[i] for i in range(3)]
        noisy_orn = p.getQuaternionFromEuler(noisy_euler)
        
        # Reset gripper with noise
        self.gripper.reset(noisy_pos, noisy_orn)
        time.sleep(0.5)  # Allow physics to settle
        
        return noisy_pos, noisy_orn

    def run_trials(self):
        for trial in range(self.num_trials):
            print(f"\nStarting trial {trial + 1}/{self.num_trials}")
            
            # Reset simulation with noise
            position, orientation = self.reset_for_next_trial()
            
            # Attempt grasp
            success = self.attempt_grasp(position, orientation)
            self.results.append(success)

if __name__ == "__main__":
    simulator = GraspSimulator(num_trials=10)
    simulator.run_trials()
    p.disconnect()