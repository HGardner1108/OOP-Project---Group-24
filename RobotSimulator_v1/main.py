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
        self.default_joint_positions = [0.05] * self.num_joints
        self.open = False
        self.constraint_id = None  # ID of the temporary constraint
        self.mass = p.getDynamicsInfo(self.gripper_id, -1)[0]
    
        # Enable friction for interaction
        self.enable_friction()
    
        # Set default joint positions
        for joint_index, joint_position in enumerate(self.default_joint_positions):
            p.resetJointState(self.gripper_id, joint_index, joint_position)
            
    def load_gripper(self):
        """Load the three-fingered gripper URDF."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "Robots/grippers/threeFingers/sdh/sdh.urdf")
        if not os.path.exists(path):
            raise FileNotFoundError(f"URDF file not found: {path}")

        initial_position = [0, 0, 0.21]
        initial_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        gripper_id = p.loadURDF(path, initial_position, initial_orientation, globalScaling=1, useFixedBase=False)
        return gripper_id
    
    def add_temporary_lock(self):
        """Lock the gripper in place by adding a temporary constraint."""
        position, orientation = p.getBasePositionAndOrientation(self.gripper_id)
        self.constraint_id = p.createConstraint(
            self.gripper_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            position,
            childFrameOrientation=orientation,
        )
        if self.constraint_id is None:
            raise RuntimeError("Failed to create a temporary lock for the gripper.")
        
    def remove_temporary_lock(self):
        """Remove the temporary constraint locking the gripper."""
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None


    def reset(self, position, orientation):
        # Override the given position and orientation with your desired values
        desired_position = [0, 0, 0.21]
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
                p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL,
                                        targetPosition=0.4, maxVelocity=2,force=1)
            done = True
        self.open = False
        self.open = False
    def close_gripper(self):
        """Close the gripper using the same logic as the testHand snippet, but ensure joints settle."""
        done = False
        while not done:
            for i in [1,4]:
                p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL, 
                                        targetPosition=0.05, maxVelocity=1, force=1)
            p.setJointMotorControl2(self.gripper_id, 7, p.POSITION_CONTROL,
                                    targetPosition=0.05, maxVelocity=1, force=2)
            done = True
            done = True
            self.add_temporary_lock()
            for joint_index in [1, 4]:
                p.setJointMotorControl2(self.gripper_id, joint_index, p.POSITION_CONTROL,
                                        targetPosition=0.05, maxVelocity=1, force=self.grip_force)
            p.setJointMotorControl2(self.gripper_id, 7, p.POSITION_CONTROL,
                                    targetPosition=0.05, maxVelocity=1, force=self.grip_force)
            for _ in range(480):
                p.stepSimulation()
                time.sleep(1/240.0)
            self.open = False
            self.remove_temporary_lock()
        # After setting targets, step the simulation a bit to let the joints move
        for _ in range(480):  # about 2 seconds at 240 Hz
            p.stepSimulation()
            time.sleep(1/240.0)

        self.open = False
    def getJointPosition(self):
        joints = []
        for i in range(self.num_joints):
            pos = p.getJointState(self.gripper_id, i)[0]
            joints.append(pos)
        return joints
        return joints
    def openGripper(self):
        closed = True
        self.add_temporary_lock()
        iteration = 0
        while(closed and not self.open):
            joints = self.getJointPosition()
            closed = False
            for k in range(self.num_joints):
                goal = 0.9
                if k in [2,5,8] and joints[k] >= goal:
                    p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
                elif k in [3,6,9] and joints[k] <= goal:
                    p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
                elif k in [1,4,7] and joints[k] <= goal:
                    p.setJointMotorControl2(self.gripper_id, k, p.POSITION_CONTROL,
                                            targetPosition=joints[k] - 0.05,
                                            maxVelocity=2,force=5)
                    closed = True
            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
        self.open = True
        self.remove_temporary_lock()
        self.remove_temporary_lock()
    
    def enable_friction(self):
        """Enable lateral friction for each of the gripper's finger links."""
        for link in range(self.num_joints):
            p.changeDynamics(self.gripper_id, link, lateralFriction=1.0)

    def lift_gripper(self, velocity=0.1, duration=3.0):
        """Lifts the gripper using position control over the specified duration."""
        # Get the current position and orientation
        current_pos, current_orn = p.getBasePositionAndOrientation(self.gripper_id)
        
        # Calculate target position
        target_height = current_pos[2] + velocity * duration
        target_pos = list(current_pos)
        target_pos[2] = target_height  # Update Z coordinate

        # Number of simulation steps
        time_step = 1./240.  # Adjust according to your simulation time step
        steps = int(duration / time_step)
        z_positions = np.linspace(current_pos[2], target_height, steps)

        # Move the gripper incrementally
        for z in z_positions:
            p.resetBasePositionAndOrientation(self.gripper_id, [current_pos[0], current_pos[1], z], current_orn)
            p.stepSimulation()
            time.sleep(time_step)
        print(f"Gripper lifted to height: {target_height}")
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
        
        # Load the plane
        self.plane_id = p.loadURDF("plane.urdf")
        print("Plane added to the simulation.")

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
        self.wait_for_enter("Press ENTER to reset the gripper to the given position/orientation...")
        self.gripper.reset(position, orientation)

        self.wait_for_enter("Press ENTER to open the gripper (it was closed initially)...")
        self.gripper.openGripper()

        self.wait_for_enter("Press ENTER to reset the block...")
        self.block.reset()

        self.wait_for_enter("Press ENTER to close (grasp) the gripper...")
        self.gripper.close_gripper()

        self.wait_for_enter("Press ENTER to lift the gripper...")
        self.gripper.lift_gripper(velocity=0.1, duration=3.0)  # Example: 0.1 m/s for 3 seconds

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
