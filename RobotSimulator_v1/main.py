import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import csv
from abc import ABC, abstractmethod

# ========== Abstract Gripper Class ==========
class Gripper(ABC):
    """Abstract base class for different types of grippers."""

    @abstractmethod
    def load_gripper(self):
        pass

    @abstractmethod
    def preshape_gripper(self):
        pass

    @abstractmethod
    def close_gripper(self):
        pass

    @abstractmethod
    def reset(self, position, orientation):
        pass

    @abstractmethod
    def lift_gripper(self, target_lift_position, orientation, num_steps=100):
        pass

    @abstractmethod
    def openGripper(self):
        pass


# ========== TwoFingerGripper ==========
class TwoFingerGripper(Gripper):
    """Two-pronged gripper."""
    def __init__(self, grip_force=1):
        self.grip_force = grip_force
        self.default_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])  # Face downwards
        self.default_joint_positions = [0.550569, 0.0, 0.549657, 0.0]  # Reference joint positions
        self.gripper_id = self.load_gripper()
        self.num_joints = p.getNumJoints(self.gripper_id)
        self.open = True  # start as open by default after reset

        # Disable gravity for the gripper
        p.changeDynamics(self.gripper_id, -1, mass=0, linearDamping=0, angularDamping=0)

    def load_gripper(self):
        """Load the two-pronged gripper URDF."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gripper_id = p.loadURDF("pr2_gripper.urdf", [0, 0, 0], self.default_orientation,
                                globalScaling=1, useFixedBase=False)
        # Set initial joint positions
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(gripper_id, joint_index, joint_position)
        print("Two-pronged gripper loaded with initial joint positions.")
        return gripper_id

    def preshape_gripper(self):
        """(Low-level) Move the gripper fingers into a preshape configuration."""
        for joint in [0, 2]:  # PR2 gripper's fingers
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.4, maxVelocity=2, force=self.grip_force)
        p.stepSimulation()
        time.sleep(1.0)

    def close_gripper(self):
        """(Low-level) Close the gripper."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=1, force=self.grip_force)
        for _ in range(240):  # Simulate for 1 second
            p.stepSimulation()
            time.sleep(1 / 240.0)
        self.open = False

    def reset(self, position, orientation):
        """Reset the gripper's position, orientation, and joint positions."""
        p.resetBasePositionAndOrientation(self.gripper_id, position, orientation)
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, joint_position)
        # After resetting, open the gripper so it's ready for next action
        self.openGripper()

    def lift_gripper(self, target_lift_position, orientation, num_steps=100):
        """Lift the gripper to a specified position."""
        if not self.open:
            for joint in [0, 2]:
                p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                        targetPosition=0.0, maxVelocity=1, force=self.grip_force)

        current_position, _ = p.getBasePositionAndOrientation(self.gripper_id)
        for step in range(num_steps):
            interpolated_position = [
                current_position[0] + (target_lift_position[0] - current_position[0]) * step / num_steps,
                current_position[1] + (target_lift_position[1] - current_position[1]) * step / num_steps,
                current_position[2] + (target_lift_position[2] - current_position[2]) * step / num_steps,
            ]
            p.resetBasePositionAndOrientation(self.gripper_id, interpolated_position, orientation)

            if not self.open:
                for joint in [0, 2]:
                    p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                            targetPosition=0.0, maxVelocity=1, force=self.grip_force)

            p.stepSimulation()
            time.sleep(1 / 240.0)
        time.sleep(1.0)

    def openGripper(self):
        """Open the gripper more fully."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.9, maxVelocity=2, force=self.grip_force)
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1/240.0)
        self.open = True


# ========== ThreeFingerGripper ==========
class ThreeFingerGripper(Gripper):
    """A three-fingered gripper."""
    def __init__(self, grip_force=1):
        self.grip_force = grip_force
        self.gripper_id = self.load_gripper()
        self.num_joints = p.getNumJoints(self.gripper_id)
        self.robot_model = self.gripper_id

        # Start as closed by default:
        # Set default_joint_positions to a closed configuration similar to what close_gripper sets:
        # close_gripper sets joints [1,4,7] to 0.05 and also joint 7. Let's just set all to 0.05 for simplicity
        self.default_joint_positions = [0.05]*self.num_joints  
        self.open = False  # start closed

        p.changeDynamics(self.gripper_id, -1, mass=1, linearDamping=0, angularDamping=0)

        # Set joints to default closed state right after loading
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, joint_position)

    def load_gripper(self):
        """Load the three-fingered gripper URDF without changing orientation for now."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "Robots/grippers/threeFingers/sdh/sdh.urdf")
        if not os.path.exists(path):
            raise FileNotFoundError(f"URDF file not found: {path}")

        # Keep the existing position and orientation as is. 
        # If you want exactly the same as the testHand code, don't change them here.
        initial_position = [0, 0, 0]  
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0]) 

        gripper_id = p.loadURDF(path, initial_position, initial_orientation, globalScaling=1, useFixedBase=False)
        print("Three-fingered gripper loaded in existing position/orientation and starting closed.")
        return gripper_id

    def reset(self, position, orientation):
        # Override the given position and orientation with your desired values
        desired_position = [0, 0, 0.1]
        desired_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        p.resetBasePositionAndOrientation(self.gripper_id, desired_position, desired_orientation)
        # Reset joints to default positions
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, joint_position)

    def preshape_gripper(self):
        # Same as before:
        done = False
        while not done:
            for i in [2,5,8]:
                p.setJointMotorControl2(self.robot_model, i, p.POSITION_CONTROL,
                                        targetPosition=0.4, maxVelocity=2,force=1)
            done = True
        self.open = False

    def close_gripper(self):
        """Close the gripper using the same logic as the testHand snippet, but ensure joints settle."""
        done = False
        while not done:
            for i in [1,4]:
                p.setJointMotorControl2(self.robot_model, i, p.POSITION_CONTROL, 
                                        targetPosition=0.05, maxVelocity=1, force=1)
            p.setJointMotorControl2(self.robot_model, 7, p.POSITION_CONTROL,
                                    targetPosition=0.05, maxVelocity=1, force=2)
            done = True

        # After setting targets, step the simulation a bit to let the joints move
        for _ in range(480):  # about 2 seconds at 240 Hz
            p.stepSimulation()
            time.sleep(1/240.0)

        self.open = False

    def getJointPosition(self):
        joints = []
        for i in range(self.num_joints):
            pos = p.getJointState(self.robot_model, i)[0]
            joints.append(pos)
        return joints

    def openGripper(self):
        closed = True
        iteration = 0
        while(closed and not self.open):
            joints = self.getJointPosition()
            closed = False
            for k in range(self.num_joints):
                goal = 0.9
                if k in [2,5,8] and joints[k] >= goal:
                    p.setJointMotorControl2(self.robot_model, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
                elif k in [3,6,9] and joints[k] <= goal:
                    p.setJointMotorControl2(self.robot_model, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
                elif k in [1,4,7] and joints[k] <= goal:
                    p.setJointMotorControl2(self.robot_model, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
        self.open = True

    def lift_gripper(self, target_lift_position, orientation, num_steps=100):
        """Lift the gripper to a specified position without changing orientation."""
        # Get the current base orientation of the gripper and do not change it
        current_position, current_orientation = p.getBasePositionAndOrientation(self.gripper_id)

        # If the gripper is closed, re-apply the close force targets during lift
        close_targets = [(1, 0.05), (4,0.05), (7,0.05)] if not self.open else []

        for step in range(num_steps):
            interpolated_position = [
                current_position[0] + (target_lift_position[0] - current_position[0]) * step / num_steps,
                current_position[1] + (target_lift_position[1] - current_position[1]) * step / num_steps,
                current_position[2] + (target_lift_position[2] - current_position[2]) * step / num_steps,
            ]

            # Notice we only use current_orientation, ignoring any passed-in orientation
            p.resetBasePositionAndOrientation(self.gripper_id, interpolated_position, current_orientation)

            if not self.open:
                for (joint_idx, target_pos) in close_targets:
                    p.setJointMotorControl2(self.robot_model, joint_idx, p.POSITION_CONTROL,
                                            targetPosition=target_pos, maxVelocity=1, force=2)

            p.stepSimulation()
            time.sleep(1 / 240.0)
    time.sleep(1.0)



# ========== Abstract Block Class ==========
class Block(ABC):
    """Abstract base class for different types of blocks."""

    @abstractmethod
    def load_block(self):
        pass

    @abstractmethod
    def reset(self):
        pass

# ========== Cube Block ==========
class Cube(Block):
    """Simple cube object."""
    def __init__(self, initial_position=None, initial_orientation=None):
        if initial_position is None:
            initial_position = [0, 0, 0.025]
        if initial_orientation is None:
            initial_orientation = [0, 0, 0, 1]
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.block_id = None
        self.load_block()

    def load_block(self):
        """Load the cube URDF."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "cube_small.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.block_id = p.loadURDF(urdf_path, self.initial_position, self.initial_orientation)
        p.changeDynamics(self.block_id, -1, mass=0.1, linearDamping=0.01, angularDamping=0.01)

    def reset(self):
        """Reset the block to its initial state."""
        p.resetBasePositionAndOrientation(self.block_id, self.initial_position, self.initial_orientation)
        p.changeDynamics(self.block_id, -1, mass=0.1, linearDamping=0.01, angularDamping=0.01)


# ========== Grasp Simulator ==========
class GraspSimulator:
    """Simulates grasping trials."""
    def __init__(self, num_trials=50, grip_force=100, gripper_type="ThreeFinger"):
        self.num_trials = num_trials
        self.grip_force = grip_force
        self.gripper_type = gripper_type
        self.gripper = None
        self.block = None
        self.results = []

        self.setup_simulation()
        self.gripper = self.create_gripper(self.grip_force, self.gripper_type)
        self.block = Cube()

    def setup_simulation(self):
        p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def create_gripper(self, grip_force, gripper_type):
        if gripper_type == "TwoFinger":
            return TwoFingerGripper(grip_force)
        elif gripper_type == "ThreeFinger":
            return ThreeFingerGripper(grip_force)
        else:
            raise ValueError(f"Unknown gripper type: {gripper_type}")

    def generate_random_pose(self, height=0.3, max_angle=np.pi / 12):
        """Generate a random pose for the gripper around the block."""
        position_noise = np.random.uniform(-0.02, 0.02, size=2)
        random_position = [
            self.block.initial_position[0] + position_noise[0],
            self.block.initial_position[1] + position_noise[1],
            self.block.initial_position[2] + height
        ]
        roll = np.random.uniform(-max_angle, max_angle)
        pitch = np.random.uniform(-max_angle, max_angle) + np.pi / 2
        yaw = np.random.uniform(-max_angle, max_angle) + np.pi
        random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        return random_position, random_orientation

    def wait_for_enter(self, message):
        input(message)
        time.sleep(3)  # Wait 3 seconds after pressing ENTER to give time to observe

    def attempt_grasp(self, position, orientation, lift_height=0.3):
        """Attempt a grasp and check if successful."""

        # Step 1: Reset the gripper
        self.gripper.reset(position, orientation)

        # Step 2: Open the gripper
        self.gripper.openGripper()

        # Step 3: Reset the block
        
        self.block.reset()
        
        # Stop block from falling due to no floor
        
        block_constraint = p.createConstraint(self.block.block_id, -1, -1, -1, 
                                      p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

        # Step 4: Close the gripper
        self.gripper.close_gripper()
        

        # Step 5: Lift the gripper
        lift_position = [position[0], position[1], position[2] + lift_height]
        self.gripper.lift_gripper(lift_position, orientation)

        # Check block position to determine success
        initial_position = p.getBasePositionAndOrientation(self.block.block_id)[0]
        time.sleep(1.0)
        final_position = p.getBasePositionAndOrientation(self.block.block_id)[0]

        if final_position[2] < initial_position[2] - 0.01:
            print("Block slipped.")
            return False

        print("Grasp successful!")
        return True


    def run_trials(self):
        """Run multiple grasp trials."""
        for i in range(self.num_trials):
            random_position, random_orientation = self.generate_random_pose()
            success = self.attempt_grasp(random_position, random_orientation)
            self.results.append({
                "trial": i + 1,
                "position": random_position,
                "orientation": random_orientation,
                "success": success
            })
            print(f"Trial {i + 1}: {'Success' if success else 'Failure'}")
        self.export_results()

    def export_results(self):
        """Export trial results to a CSV file."""
        csv_file = "grasp_results.csv"
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["trial", "position", "orientation", "success"])
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        print(f"Results exported to {csv_file}")

    def run(self):
        """Run the simulation."""
        self.run_trials()
        p.disconnect()


if __name__ == "__main__":
    simulator = GraspSimulator(num_trials=50, grip_force=100, gripper_type="ThreeFinger")
    simulator.run()
