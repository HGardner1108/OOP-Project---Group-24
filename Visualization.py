import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import numpy as np
from scipy.spatial.transform import Rotation as R

# read data
data = pd.read_csv("RobotSimulator_v1/grasp_results.csv")

data['position'] = data['position'].str.replace("np.float64", "").str.replace(" ", "").apply(ast.literal_eval)
data['orientation'] = data['orientation'].str.replace("np.float64", "").str.replace(" ", "").apply(ast.literal_eval)

data[['X', 'Y', 'Z']] = pd.DataFrame(data['position'].tolist(), index=data.index)
data[['qx', 'qy', 'qz', 'qw']] = pd.DataFrame(data['orientation'].tolist(), index=data.index)


def quaternion_to_vector(x, y, z, w):
    quat = R.from_quat([x, y, z, w])
    vector = quat.apply([1, 0, 0])
    #print(vector)
    #print(quat)
    return vector


success = data[data['success'] == 1]
failure = data[data['success'] == 0]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(success['X'], success['Y'], success['Z'], c='green', label='Success', alpha=0.6)
ax.scatter(failure['X'], failure['Y'], failure['Z'], c='red', label='Failure', alpha=0.6)

for _, row in data.iterrows():
    x, y, z = row['X'], row['Y'], row['Z']
    qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']

    direction = quaternion_to_vector(qx, qy, qz, qw)
    ax.quiver(x, y, z, direction[0], direction[1], direction[2], length=qw, color='orange', alpha=0.5)

cube_center = [0, 0, 0.025]
cube_side = 0.05
half_side = cube_side / 2

vertices = np.array([
    [cube_center[0] - half_side, cube_center[1] - half_side, cube_center[2] - half_side],  # Bottom-front-left
    [cube_center[0] + half_side, cube_center[1] - half_side, cube_center[2] - half_side],  # Bottom-front-right
    [cube_center[0] + half_side, cube_center[1] + half_side, cube_center[2] - half_side],  # Bottom-back-right
    [cube_center[0] - half_side, cube_center[1] + half_side, cube_center[2] - half_side],  # Bottom-back-left
    [cube_center[0] - half_side, cube_center[1] - half_side, cube_center[2] + half_side],  # Top-front-left
    [cube_center[0] + half_side, cube_center[1] - half_side, cube_center[2] + half_side],  # Top-front-right
    [cube_center[0] + half_side, cube_center[1] + half_side, cube_center[2] + half_side],  # Top-back-right
    [cube_center[0] - half_side, cube_center[1] + half_side, cube_center[2] + half_side]   # Top-back-left
])

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

for edge in edges:
    ax.plot(
        [vertices[edge[0], 0], vertices[edge[1], 0]],
        [vertices[edge[0], 1], vertices[edge[1], 1]],
        [vertices[edge[0], 2], vertices[edge[1], 2]],
        color='blue'
    )


def set_equal_aspect(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


set_equal_aspect(ax)

ax.set_title('Gripper Visualisation')
ax.set_xlabel('Gripper X')
ax.set_ylabel('Gripper Y')
ax.set_zlabel('Gripper Z')
ax.legend()

plt.show()
