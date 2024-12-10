import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class GraspClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.pca = None

    def load_data(self, csv_file):
        """Load and preprocess data from CSV file."""
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Extract positions and orientations from string representations
        positions = df['position'].apply(eval)
        orientations = df['orientation'].apply(eval)
        
        # Create feature matrix X
        X = np.zeros((len(df), 7))  # 3 for position, 4 for quaternion
        
        # Extract position coordinates
        X[:, 0] = positions.apply(lambda x: x[0])  # x
        X[:, 1] = positions.apply(lambda x: x[1])  # y
        X[:, 2] = positions.apply(lambda x: x[2])  # z
        
        # Extract quaternion components
        X[:, 3] = orientations.apply(lambda x: x[0])  # qx
        X[:, 4] = orientations.apply(lambda x: x[1])  # qy
        X[:, 5] = orientations.apply(lambda x: x[2])  # qz
        X[:, 6] = orientations.apply(lambda x: x[3])  # qw
        
        # Get labels
        y = df['success'].values
        
        return X, y

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        return self.classifier.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        return report, matrix

    def cross_validate(self, X, y, n_splits=5):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.classifier.fit(X_train, y_train)
            scores.append(self.classifier.score(X_val, y_val))
        return np.mean(scores), np.std(scores)

    def feature_importances(self):
        return self.classifier.feature_importances_

    def apply_pca(self, X_train, X_test, variance_ratio=0.95):
        self.pca = PCA(n_components=variance_ratio)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        return X_train_pca, X_test_pca

    def data_size_performance(self, X_train, y_train, X_test, y_test, train_sizes):
        performances = []
        for size in train_sizes:
            subset_size = int(size * len(X_train))
            X_subset, y_subset = X_train[:subset_size], y_train[:subset_size]
            self.classifier.fit(X_subset, y_subset)
            score = self.classifier.score(X_test, y_test)
            performances.append(score)
        return performances

    def plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve using test data."""
        # Get probabilities for positive class
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
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

