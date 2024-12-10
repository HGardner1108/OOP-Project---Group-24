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
import os
from skorch import NeuralNetBinaryClassifier

# Ensure plots directory exists
os.makedirs('2x_plots', exist_ok=True)

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

def main():
    # Load data
    X, y = load_data('2x_normalised_grasp_results.csv')

    # Split into training and testing sets
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize the model, loss function, and optimizer using Skorch
    net = Net()
    net = NeuralNetBinaryClassifier(
        module=Net,
        max_epochs=200,
        lr=0.001,
        batch_size=16,
        optimizer=optim.Adam,
        criterion=nn.BCELoss,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )

    # Fit the model
    net.fit(X_train_np.astype(np.float32), y_train_np.astype(np.float32))

    # Evaluate on test set
    y_pred = net.predict(X_test_np.astype(np.float32))
    y_proba = net.predict_proba(X_test_np.astype(np.float32))[:, 1]

    accuracy = accuracy_score(y_test_np, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test_np, y_pred))

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test_np, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
    plot_roc_curve_fn(fpr, tpr, roc_auc, '2x_plots/roc_curve.png')
    print("ROC curve saved to '2x_plots/roc_curve.png'.")

    # Compute Confusion Matrix
    cm = confusion_matrix(y_test_np, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Plot and save Confusion Matrix
    plot_confusion_matrix_fn(cm, ['Failure', 'Success'], '2x_plots/confusion_matrix.png')
    print("Confusion matrix saved to '2x_plots/confusion_matrix.png'.")

    # Feature Importance via Permutation
    print("Calculating feature importances via permutation...")
    feature_names = ['X', 'Y', 'Z', 'roll', 'pitch', 'yaw']
    # Using sklearn's permutation_importance with Skorch's compatible estimator
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        net, X_test_np.astype(np.float32), y_test_np, n_repeats=10, random_state=42, scoring='accuracy'
    )
    feature_importances = result.importances_mean
    print("Feature importances calculated.")

    # Plot and save Feature Importances
    plot_feature_importance_fn(feature_importances, feature_names, '2x_plots/feature_importances.png')
    print("Feature importances plot saved to '2x_plots/feature_importances.png'.")

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_np)
    print(f"PCA completed. Explained variance ratios: {pca.explained_variance_ratio_}")

    # Plot and save PCA Explained Variance
    plot_pca_variance_fn(pca, '2x_plots/pca_variance.png')
    print("PCA explained variance plot saved to '2x_plots/pca_variance.png'.")

    # Plot and save Learning Curve using Skorch's compatible estimator
    print("Generating learning curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=net,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )

    plot_learning_curve_fn(train_sizes, train_scores, test_scores, '2x_plots/learning_curve.png')
    print("Learning curve saved to '2x_plots/learning_curve.png'.")

    print("All plots have been saved to the '2x_plots' directory.")

if __name__ == "__main__":
    main()