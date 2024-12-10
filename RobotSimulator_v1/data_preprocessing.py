import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('grasp_results.csv')

# Parse 'position' and 'orientation' columns
data['position'] = data['position'].apply(eval)
data['orientation'] = data['orientation'].apply(eval)

# Split 'position' into X, Y, Z
data[['X', 'Y', 'Z']] = pd.DataFrame(data['position'].tolist(), index=data.index)

# Split 'orientation' into qx, qy, qz, qw
data[['qx', 'qy', 'qz', 'qw']] = pd.DataFrame(data['orientation'].tolist(), index=data.index)

# Identify the minority class
success_count = data['success'].sum()
failure_count = len(data) - success_count

minority_class = True if success_count < failure_count else False
minority_size = min(success_count, failure_count)

# Separate classes
success_cases = data[data['success'] == True]
failure_cases = data[data['success'] == False]

# Balance the dataset
if minority_class:
    balanced_success = success_cases
    balanced_failure = failure_cases.sample(n=minority_size, random_state=42)
else:
    balanced_failure = failure_cases
    balanced_success = success_cases.sample(n=minority_size, random_state=42)

balanced_data = pd.concat([balanced_success, balanced_failure]).reset_index(drop=True)

# Convert quaternion to roll, pitch, yaw
def quaternion_to_euler(qx, qy, qz, qw):
    quat = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = quat.as_euler('xyz', degrees=False)
    return roll, pitch, yaw

euler_angles = balanced_data.apply(
    lambda row: quaternion_to_euler(row['qx'], row['qy'], row['qz'], row['qw']), axis=1)
balanced_data[['roll', 'pitch', 'yaw']] = pd.DataFrame(euler_angles.tolist(), index=balanced_data.index)

# Select features to normalize
features = ['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']

# Normalize the features
scaler = StandardScaler()
balanced_data[features] = scaler.fit_transform(balanced_data[features])

# The balanced and preprocessed data is now ready for use

# Shuffle the rows in the DataFrame
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder columns as per your requirement
columns_to_save = ['X', 'Y', 'Z', 'roll', 'pitch', 'yaw', 'success']
balanced_data = balanced_data[columns_to_save]

# Save the DataFrame to a CSV file
balanced_data.to_csv('normalised_grasp_results.csv', index=False)