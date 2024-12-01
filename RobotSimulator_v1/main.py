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
        self.gripper_id = self.load_gripper()
        self.num_joints = p.getNumJoints(self.gripper_id)

    def load_gripper(self):
        """Load the pr2 gripper URDF."""
        # Set search path for PyBullet's data URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gripper_id = p.loadURDF("pr2_gripper.urdf", [0, 0, 0], self.default_orientation,
                                globalScaling=1, useFixedBase=False)
        print("Two-pronged gripper loaded.")
        return gripper_id

    def open_gripper(self):
        """Open the gripper to its fully open position."""
        for joint in [0, 2]:  # PR2 gripper's fingers
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL, 
                                    targetPosition=0.0, maxVelocity=1, force=self.grip_force)
        p.stepSimulation()
        time.sleep(0.5)

    def close_gripper(self):
        """Close the gripper."""
        for joint in [0, 2]:  # PR2 gripper's fingers
            p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL, 
                                    targetPosition=0.1, maxVelocity=1, force=self.grip_force)
        p.stepSimulation()
        time.sleep(0.5)

    def reset(self, position, orientation):
        """Reset the gripper's position and orientation."""
        p.resetBasePositionAndOrientation(self.gripper_id, position, orientation)
        self.open_gripper()

class Block:
    """Represents the block object."""
    def __init__(self, initial_position=None, initial_orientation=None):
        if initial_position is None:
            initial_position = [0, 0, 0.1]
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
    def __init__(self, num_trials=50, grip_force=1):
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

    def generate_random_pose(self, height=0.35, max_angle= np.pi / 12):
        """Generate a random pose for the gripper around the block."""
        position_noise = np.random.uniform(-0.02, 0.02, size=2)
        random_position = [
            self.block.initial_position[0] + position_noise[0],
            self.block.initial_position[1] + position_noise[1],
            self.block.initial_position[2] + height
        ]
        roll = np.random.uniform(-max_angle, max_angle)
        pitch = np.random.uniform(-max_angle, max_angle) + np.pi/2
        yaw = np.random.uniform(-max_angle, max_angle) + np.pi
        random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        return random_position, random_orientation

    def attempt_grasp(self, position, orientation, lift_height=0.15, wait_time=3):
        """Attempt a grasp and check if successful."""
        # Reset gripper and block
        self.gripper.reset(position, orientation)
        self.block.reset()

        # Close gripper
        self.gripper.close_gripper()

        # Check for contact
        contact_points = p.getContactPoints(bodyA=self.gripper.gripper_id, bodyB=self.block.block_id)
        if not contact_points:
            print("No contact detected. Skipping pose.")
            return False

        # Enable gravity for the block
        p.changeDynamics(self.block.block_id, -1, mass=0.1, linearDamping=0.01, angularDamping=0.01)

        # Lift the block
        lift_position = [position[0], position[1], position[2] + lift_height]
        num_steps = 50
        for step in range(num_steps):
            interpolated_position = [
                position[0],
                position[1],
                position[2] + (lift_height * step / num_steps)
            ]
            p.resetBasePositionAndOrientation(self.gripper.gripper_id, interpolated_position, orientation)
            p.stepSimulation()
            time.sleep(1 / 240.0)

        # Monitor for slipping
        start_time = time.time()
        initial_position = p.getBasePositionAndOrientation(self.block.block_id)[0]
        while time.time() - start_time < wait_time:
            p.stepSimulation()
            time.sleep(1 / 240.0)
            current_position = p.getBasePositionAndOrientation(self.block.block_id)[0]
            if current_position[2] < initial_position[2] - 0.01:
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
    simulator = GraspSimulator(num_trials=50, grip_force=1)
    simulator.run()
