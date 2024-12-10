RobotSimulator
RobotSimulator is a Python-based simulation environment for robotic grasping tasks using the PyBullet physics engine. This project includes various modules for simulating different types of robotic grippers and blocks, running grasping trials, and analyzing the results.

Project Structure
Installation
Clone the repository:

Set up a virtual environment:

Install the required dependencies:

Usage
Running the Simulation
To run the main simulation script, execute:

This script will run grasping trials for both small and large cubes using a three-finger gripper and save the results to CSV files.

Running Tests
To run the tests, execute:

Project Components
Main Simulation
main.py: The main script to run grasping trials. It initializes the simulation environment, runs trials, and saves the results.
Blocks
Cube: Represents a small cube block.
CubeLarge: Represents a large cube block.
Gripper
ThreeFingerGripper: A class representing a three-finger robotic gripper.

Plotting
plotting.py: Contains functions for visualizing the results of the grasping trials. This includes 3D scatter plots and orientation vector plots.
Example Usage:
Neural Network Classifier
NN_Classifier.py: Implements a neural network classifier for predicting the success of grasp attempts based on the collected data.
Example Usage:

'''
from plotting_module import plot_3d_scatter_with_labels, plot_3d_with_vectors

# Define file paths
csv_file = 'grasp_results.csv'
output_file_scatter = 'scatter_plot.png'
output_file_quiver = 'quiver_plot.png'

# Plot 3D visualizations
plot_3d_scatter_with_labels(csv_file, output_file=output_file_scatter)
plot_3d_with_vectors(csv_file, output_file=output_file_quiver)
'''

Neural Network Classifier
NN_Classifier.py: Implements a neural network classifier for predicting the success of grasp attempts based on the collected data.

Example Usage:
'''
from classifier_module import GraspClassifier

# Load and preprocess the data
classifier = GraspClassifier(n_estimators=100, random_state=42)
X, y = classifier.load_data('grasp_results.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the classifier
classifier.train(X_train, y_train)

# Evaluate the classifier
report, matrix = classifier.evaluate(X_test, y_test)
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)
'''

Acknowledgments
PyBullet: Physics engine used for the simulation.
NumPy: Library for numerical computations.
For more information, please refer to the documentation or contact the project maintainers.
