import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
import os
from skorch import NeuralNetBinaryClassifier

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the neural network model with LeakyReLU and Dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(32, 16)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.leaky_relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x

def load_data(csv_file):
    """Load and preprocess data from CSV file."""
    df = pd.read_csv(csv_file)
    X = df[['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']].values
    y = df['success'].values.astype(np.float32)
    return X, y

def plot_confusion_matrix_fn(cm, classes, save_path):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve_fn(fpr, tpr, roc_auc, save_path):
    """Plot and save the ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance_fn(importances, feature_names, save_path):
    """Plot and save feature importances."""
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importances (Permutation Importance)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Decrease in Accuracy')
    plt.savefig(save_path)
    plt.close()

def plot_pca_variance_fn(pca, save_path):
    """Plot and save PCA explained variance."""
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve_fn(train_sizes, train_scores, test_scores, save_path):
    """Plot and save the learning curve."""
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_tsne(X, y, save_path, n_components=2):
    """Apply T-SNE and plot the results."""
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['green' if label == 1 else 'red' for label in y]
    if n_components == 2:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=50, alpha=0.7)
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
    elif n_components == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors, s=50, alpha=0.7)
        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_zlabel('TSNE Component 3')
    plt.title('T-SNE Visualization')
    plt.savefig(save_path)
    plt.close()

def evaluate_model_with_splits(X, y, splits, model_class, output_dir):
    """Evaluate the model with different train-test splits."""
    results = []

    for split in splits:
        print(f"\nEvaluating with train-test split: {split*100:.0f}% train / {100-split*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42, stratify=y)

        # Initialize the model, loss function, and optimizer using Skorch
        net = NeuralNetBinaryClassifier(
            module=model_class,
            max_epochs=100,
            lr=0.001,
            batch_size=16,
            optimizer=optim.Adam,
            criterion=nn.BCELoss,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0
        )

        # Fit the model
        net.fit(X_train.astype(np.float32), y_train.astype(np.float32))

        # Evaluate on test set
        y_pred = net.predict(X_test.astype(np.float32))
        y_proba = net.predict_proba(X_test.astype(np.float32))[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test Accuracy: {accuracy:.4f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Plot and save ROC curve
        roc_curve_path = os.path.join(output_dir, f'roc_curve_{int(split*100)}.png')
        plot_roc_curve_fn(fpr, tpr, roc_auc, roc_curve_path)
        print(f"ROC curve saved to '{roc_curve_path}'.")

        # Compute Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:')
        print(cm)

        # Plot and save Confusion Matrix
        cm_path = os.path.join(output_dir, f'confusion_matrix_{int(split*100)}.png')
        plot_confusion_matrix_fn(cm, ['Failure', 'Success'], cm_path)
        print(f"Confusion matrix saved to '{cm_path}'.")

        # Save results
        results.append({
            'split': split,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })

    return results

def main():
    # Load data
    X, y = load_data('normalised_grasp_results.csv')

    # Define train-test splits to evaluate
    splits = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Create output directory for results
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate model with different train-test splits
    results = evaluate_model_with_splits(X, y, splits, Net, output_dir)

    # Print summary of results
    for result in results:
        print(f"\nTrain-Test Split: {result['split']*100:.0f}% train / {100-result['split']*100:.0f}% test")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        print(f"Confusion Matrix:\n{result['confusion_matrix']}")

    # Apply T-SNE and plot the results
    tsne_path_2d = os.path.join(output_dir, 'tsne_2d.png')
    plot_tsne(X, y, tsne_path_2d, n_components=2)
    print(f"T-SNE 2D plot saved to '{tsne_path_2d}'.")

    tsne_path_3d = os.path.join(output_dir, 'tsne_3d.png')
    plot_tsne(X, y, tsne_path_3d, n_components=3)
    print(f"T-SNE 3D plot saved to '{tsne_path_3d}'.")

if __name__ == "__main__":
    main()