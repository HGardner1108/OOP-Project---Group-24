from classifier_module import GraspClassifier
from plotting_module import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_performance_vs_data_size, 
    plot_feature_importance,
    plot_3d_scatter_with_labels
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess the data
classifier = GraspClassifier(n_estimators=100, random_state=420)

# Load the grasp results
X, y = classifier.load_data('grasp_results.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply PCA to reduce dimensionality
X_train_pca, X_test_pca = classifier.apply_pca(X_train, X_test)

# Train the classifier
print("Training the Random Forest classifier...")
classifier.train(X_train_pca, y_train)

# After training, get feature names based on number of PCA components
n_components = X_train_pca.shape[1]
feature_names = [f"PC{i+1}" for i in range(n_components)]

# Evaluate the classifier
print("Evaluating the Random Forest classifier...")
report, matrix = classifier.evaluate(X_test_pca, y_test)
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

# Plot the ROC curve
classifier.plot_roc_curve(X_test_pca, y_test)

# Plot feature importance
importances = classifier.classifier.feature_importances_
plot_feature_importance(importances, feature_names)

# Define file paths
csv_file = 'grasp_results.csv'
output_file_scatter = 'scatter_plot.png'
output_file_quiver = 'quiver_plot.png'

# Plot 3D visualizations
plot_3d_scatter_with_labels(csv_file, output_file=output_file_scatter)

def plot_3d_with_vectors(csv_file, output_file='vector_plot.png', sample_size=50):
    """Plot 3D scatter with orientation vectors for sampled grasp attempts."""
    # Read data
    df = pd.read_csv(csv_file)
    
    # Sample equal numbers of successful and failed grasps
    successful = df[df['success'] == True].sample(n=sample_size, random_state=42)
    failed = df[df['success'] == False].sample(n=sample_size, random_state=42)
    
    # Combine samples
    sampled_df = pd.concat([successful, failed])
    
    # Extract positions and orientations
    positions = sampled_df['position'].apply(eval)
    orientations = sampled_df['orientation'].apply(eval)
    
    # Extract coordinates and create vectors
    gripper_x = positions.apply(lambda x: x[0])
    gripper_y = positions.apply(lambda x: x[1])
    gripper_z = positions.apply(lambda x: x[2])
    
    def quat_to_vector(quat):
        w, x, y, z = quat
        x_dir = 2 * (x*z + w*y)
        y_dir = 2 * (y*z - w*x)
        z_dir = 1 - 2 * (x*x + y*y)
        return [x_dir, y_dir, z_dir]
    
    vectors = orientations.apply(quat_to_vector)
    orientation_x = vectors.apply(lambda v: v[0])
    orientation_y = vectors.apply(lambda v: v[1])
    orientation_z = vectors.apply(lambda v: v[2])
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color based on success
    colors = ['green' if success else 'red' for success in sampled_df['success']]
    
    # Plot with smaller arrows
    ax.quiver(
        gripper_x, gripper_y, gripper_z,
        orientation_x, orientation_y, orientation_z,
        color=colors, 
        length=0.05,  # Reduced from 0.15 to 0.05
        normalize=True, 
        linewidth=0.8  # Slightly reduced linewidth
    )
    ax.scatter(gripper_x, gripper_y, gripper_z, c=colors, s=100)
    
    # Rest of the visualization code remains the same
    ax.set_xlabel('Gripper X Position (m)')
    ax.set_ylabel('Gripper Y Position (m)')
    ax.set_zlabel('Gripper Z Position (m)')
    ax.set_title(f'Sampled Grasp Attempts ({sample_size} each)')
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=10, label='Successful Grasp'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=10, label='Failed Grasp')
    ]
    ax.legend(handles=legend_elements)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

plot_3d_with_vectors(csv_file, output_file=output_file_quiver)
