import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class GraspClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.pca = None

    def load_data(self, file_path):
        """Loads and preprocesses grasp data."""
        data = pd.read_csv(file_path)
        data['position'] = data['position'].apply(eval)  # Convert string to list
        positions = pd.DataFrame(data['position'].tolist(), columns=['x', 'y', 'z'])
        orientations = pd.DataFrame(data['orientation'].apply(eval).tolist(), columns=['qx', 'qy', 'qz', 'qw'])
        X = pd.concat([positions, orientations], axis=1).values
        y = data['success'].astype(int).values
        return X, y

    def train(self, X_train, y_train):
        """Trains the random forest classifier."""
        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        """Predicts using the random forest classifier."""
        return self.classifier.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluates the random forest classifier and returns metrics."""
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        return report, matrix

    def plot_roc_curve(self, X_test, y_test):
        """Plots the ROC curve."""
        y_pred_prob = self.classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def apply_pca(self, X_train, X_test, variance_ratio=0.95):
        """Applies PCA for dimensionality reduction."""
        self.pca = PCA(n_components=variance_ratio)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        return X_train_pca, X_test_pca

    def cross_validate(self, X, y, n_splits=5):
        """Performs cross-validation to estimate the model performance."""
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.train(X_train, y_train)
            scores.append(self.classifier.score(X_val, y_val))
        return np.mean(scores), np.std(scores)

    def data_size_performance(self, X_train, y_train, X_test, y_test, train_sizes):
        """Analyzes performance with increasing data sizes."""
        performances = []
        for size in train_sizes:
            subset_size = int(size * len(X_train))
            X_subset, y_subset = X_train[:subset_size], y_train[:subset_size]
            self.train(X_subset, y_subset)
            score = self.classifier.score(X_test, y_test)
            performances.append(score)
        return performances
        
