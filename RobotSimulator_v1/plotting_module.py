# plotting_module.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter_with_labels(csv_file, output_file=None):
    """Plot 3D scatter plot of grasp positions with binary success/failure."""
    df = pd.read_csv(csv_file)
    
    # Extract positions from string representations
    positions = df['position'].apply(eval)
    
    # Extract coordinates
    x = positions.apply(lambda x: x[0])
    y = positions.apply(lambda x: x[1])
    z = positions.apply(lambda x: x[2])
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with binary colors
    colors = ['green' if success else 'red' for success in df['success']]
    ax.scatter(x, y, z, c=colors, s=50)
    
    # Set labels and title
    ax.set_xlabel('Gripper X Position (m)')
    ax.set_ylabel('Gripper Y Position (m)')
    ax.set_zlabel('Gripper Z Position (m)')
    ax.set_title('Grasp Success by Position')
    
    # Legend for binary outcomes
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Successful Grasp'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Failed Grasp')
    ]
    ax.legend(handles=legend_elements)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_3d_with_vectors(csv_file, output_file=None):
    """Plot 3D scatter with orientation vectors"""
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

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    if classes is None:
        classes = ['Failure', 'Success']
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title or 'Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_performance_vs_data_size(train_sizes, train_scores, test_scores):
    """Plot learning curves."""
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_importance, feature_names):
    """Plot feature importance with correct tick handling."""
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(feature_importance))
    plt.bar(positions, feature_importance)
    plt.xticks(positions, feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()