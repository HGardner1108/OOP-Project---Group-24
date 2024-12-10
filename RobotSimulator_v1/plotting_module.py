import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_scatter_with_labels(csv_file, output_file=None):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Extract position and labels
    gripper_x = data['GripperX']
    gripper_y = data['GripperY']
    gripper_z = data['GripperZ']
    labels = data['Label']  # Assuming 'Label' column has Success, Failure, Almost
    
    # Map labels to colors
    label_color_map = {'Success': 'green', 'Failure': 'red', 'Almost': 'blue'}
    colors = labels.map(label_color_map)

    # Plot 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gripper_x, gripper_y, gripper_z, c=colors, label=labels, s=5)

    # Set labels and title
    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('3D Gripper Visualization')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Success'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Failure'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Almost'),
    ]
    ax.legend(handles=legend_elements)

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()


def plot_3d_with_vectors(csv_file, output_file=None):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract position, orientation, and labels
    gripper_x = data['GripperX']
    gripper_y = data['GripperY']
    gripper_z = data['GripperZ']
    orientation_x = data['OrientationX']
    orientation_y = data['OrientationY']
    orientation_z = data['OrientationZ']
    labels = data['Label']  # Assuming 'Label' column has Success, Failure, Almost

    # Map labels to colors
    label_color_map = {'Success': 'green', 'Failure': 'red', 'Almost': 'blue'}
    colors = labels.map(label_color_map)

    # Plot 3D scatter with vectors
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(
        gripper_x, gripper_y, gripper_z,  # Starting point of vectors
        orientation_x, orientation_y, orientation_z,  # Vector components
        color=colors, length=0.1, normalize=True, linewidth=0.5
    )
    ax.scatter(gripper_x, gripper_y, gripper_z, c=colors, s=5)

    # Set labels and title
    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('3D Gripper Visualization with Vectors')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Success'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Failure'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Almost'),
    ]
    ax.legend(handles=legend_elements)

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
