import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import csv

class TwoProngGripper:
    """Represents the two-pronged gripper."""
    def __init__(self, grip_force=1):
        self.grip_force = grip_force
        self.default_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])  # Face downwards
        self.default_joint_positions = [0.550569, 0.0, 0.549657, 0.0]  # Reference joint positions
        self.gripper_id = self.load_gripper()
        self.num_joints = p.getNumJoints(self.gripper_id)

        # Disable gravity for the gripper
        p.changeDynamics(self.gripper_id, -1, mass=0, linearDamping=0, angularDamping=0)

    def load_gripper(self):
        """Load the PR2 gripper URDF."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gripper_id = p.loadURDF("pr2_gripper.urdf", [0, 0, 0], self.default_orientation,
                                globalScaling=1, useFixedBase=False)
        # Set initial joint positions
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(gripper_id, joint_index, joint_position)
        print("Two-pronged gripper loaded with initial joint positions.")
        return gripper_id

    def preshape_gripper(self):
        """Move the gripper fingers into a preshape configuration."""
        for joint in [0, 2]:  # PR2 gripper's fingers
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.4, maxVelocity=2, force=self.grip_force)
        p.stepSimulation()
        time.sleep(1.0)  # Allow the preshape to complete

    def close_gripper(self):
        """Close the gripper by setting joints [0, 2] explicitly to position 0."""
        for joint in [0, 2]:  # PR2 gripper's fingers
            # Explicitly reset the joint state to 0.0
            p.resetJointState(self.gripper_id, joint, 0.0)
        p.stepSimulation()
        time.sleep(1.0)  # Allow the gripper to settle in the closed state

    def reset(self, position, orientation):
        """Reset the gripper's position, orientation, and joint positions."""
        p.resetBasePositionAndOrientation(self.gripper_id, position, orientation)
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, joint_position)
        self.preshape_gripper()

    def lift_gripper(self, target_lift_position, orientation, num_steps=100):
        """Lift the gripper to a specified position while maintaining orientation."""

        # Lock gripper joint positions
        for joint in [0, 2]:  # PR2 gripper's fingers
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=1, force=self.grip_force)

        # Get the current position of the gripper
        current_position, _ = p.getBasePositionAndOrientation(self.gripper_id)

        # Lift gripper from the current position to the target position
        for step in range(num_steps):
            interpolated_position = [
                current_position[0] + (target_lift_position[0] - current_position[0]) * step / num_steps,
                current_position[1] + (target_lift_position[1] - current_position[1]) * step / num_steps,
                current_position[2] + (target_lift_position[2] - current_position[2]) * step / num_steps,
            ]
            p.resetBasePositionAndOrientation(self.gripper_id, interpolated_position, orientation)
            p.stepSimulation()
            time.sleep(1 / 240.0)





class Block:
    """Represents the block object."""
    def __init__(self, initial_position=None, initial_orientation=None):
        if initial_position is None:
            initial_position = [0, 0, 0.02]
        if initial_orientation is None:
            initial_orientation = [0, 0, 0, 1]
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.block_id = None  # Initialize block_id
        self.load_block()

    def load_block(self):
        """Load the block URDF and configure its dynamics."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "cube_small.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.block_id = p.loadURDF(urdf_path, self.initial_position, self.initial_orientation)
        self.configure_dynamics()  # Call after block_id is set

    def configure_dynamics(self):
        """Configure the block's dynamics."""
        p.changeDynamics(self.block_id, -1, mass=0.1, linearDamping=0.01, angularDamping=0.01)

    def reset(self):
        """Reset the block to its initial state."""
        p.resetBasePositionAndOrientation(self.block_id, self.initial_position, self.initial_orientation)
        self.configure_dynamics()


class GraspSimulator:
    """Simulates grasping trials."""
    def __init__(self, num_trials=50, grip_force=100):
        self.num_trials = num_trials
        self.gripper = None
        self.block = None
        self.results = []

        self.setup_simulation()
        self.gripper = TwoProngGripper(grip_force=grip_force)
        self.block = Block()

    def setup_simulation(self):
        """Set up the PyBullet simulation."""
        p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1 / 240.0)

        # Add floor
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        print("Floor added.")

    def generate_random_pose(self, height=0.29, max_angle=np.pi / 12):
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

    def attempt_grasp(self, position, orientation, lift_height=0.3):
        """Attempt a grasp and check if successful."""
        self.gripper.reset(position, orientation)
        self.block.reset()

        self.gripper.preshape_gripper()
        self.gripper.close_gripper()

        lift_position = [position[0], position[1], position[2] + lift_height]
        self.gripper.lift_gripper(lift_position, orientation)

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
    simulator = GraspSimulator(num_trials=50, grip_force=100)
    simulator.run()
