import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import numpy as np
from scipy.spatial.transform import Rotation as R

# Read data
data = pd.read_csv("grasp_results.csv")

# Clean and parse the data
data['position'] = data['position'].str.replace("np.float64", "").str.replace(" ", "").apply(ast.literal_eval)
data['orientation'] = data['orientation'].str.replace("np.float64", "").str.replace(" ", "").apply(ast.literal_eval)

data[['X', 'Y', 'Z']] = pd.DataFrame(data['position'].tolist(), index=data.index)
data[['qx', 'qy', 'qz', 'qw']] = pd.DataFrame(data['orientation'].tolist(), index=data.index)

def quaternion_to_vector(qx, qy, qz, qw):
    quat = R.from_quat([qx, qy, qz, qw])
    vector = quat.apply([0, 0, 1])
    return vector

# Sample 50 from each class
success = data[data['success'] == True].sample(n=50, random_state=42)
failure = data[data['success'] == False].sample(n=50, random_state=42)

# Combine samples
sampled_data = pd.concat([success, failure])

# Compute orientation vectors
vectors = sampled_data.apply(lambda row: quaternion_to_vector(row['qx'], row['qy'], row['qz'], row['qw']), axis=1)
sampled_data[['u', 'v', 'w']] = pd.DataFrame(vectors.tolist(), index=sampled_data.index)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define cube parameters
cube_center = [0, 0, 0.025]  # Cube center coordinates
cube_side = 0.05             # Full side length of the cube
half_side = cube_side / 2    # Half the length of cube side

# Define vertices of the cube
vertices = np.array([
    [cube_center[0] - half_side, cube_center[1] - half_side, cube_center[2] - half_side],  # Bottom-front-left
    [cube_center[0] + half_side, cube_center[1] - half_side, cube_center[2] - half_side],  # Bottom-front-right
    [cube_center[0] + half_side, cube_center[1] + half_side, cube_center[2] - half_side],  # Bottom-back-right
    [cube_center[0] - half_side, cube_center[1] + half_side, cube_center[2] - half_side],  # Bottom-back-left
    [cube_center[0] - half_side, cube_center[1] - half_side, cube_center[2] + half_side],  # Top-front-left
    [cube_center[0] + half_side, cube_center[1] - half_side, cube_center[2] + half_side],  # Top-front-right
    [cube_center[0] + half_side, cube_center[1] + half_side, cube_center[2] + half_side],  # Top-back-right
    [cube_center[0] - half_side, cube_center[1] + half_side, cube_center[2] + half_side],  # Top-back-left
])

# Define edges of the cube
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # Bottom edges
    [4, 5], [5, 6], [6, 7], [7, 4],   # Top edges
    [0, 4], [1, 5], [2, 6], [3, 7]    # Side edges
]

# Plot the cube
for edge in edges:
    ax.plot(
        [vertices[edge[0], 0], vertices[edge[1], 0]],
        [vertices[edge[0], 1], vertices[edge[1], 1]],
        [vertices[edge[0], 2], vertices[edge[1], 2]],
        color='blue'
    )

# Plot sampled data with smaller arrows colored by success
for _, row in sampled_data.iterrows():
    x, y, z = row['X'], row['Y'], row['Z']
    u, v, w = row['u'], row['v'], row['w']
    color = 'green' if row['success'] else 'red'
    ax.quiver(x, y, z, u, v, w, color=color, length=0.05, normalize=True, linewidth=0.8)
    ax.scatter(x, y, z, c=color, s=50)

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Sampled Grasp Attempts with Orientation Vectors')

# Adjust the aspect ratio to make the scales of x, y, z the same
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the data coordinate system
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Call the function to set equal aspect ratio
set_axes_equal(ax)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Success', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Failure', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements)

plt.show()
