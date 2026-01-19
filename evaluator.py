import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from config import Config

class Evaluator:
    """Evaluate classification performance for each architecture."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_features(self, split_name):
        """Load pre-computed features."""
        path = Config.FEATURES_DIR / f"{self.model_name}_{split_name}.h5"
        
        with h5py.File(path, "r") as f:
            features = f["features"][:]
            positions = f["positions"][:]
        
        return features, positions
    
    def evaluate(self):
        """Evaluate classification on train/test and cross-session generalization."""
        train_features, train_positions = self.load_features("train")
        test_features, test_positions = self.load_features("test")
        
        clf = KNeighborsClassifier(n_neighbors=Config.K_NEIGHBORS)
        clf.fit(train_features, train_positions)
        
        pred = clf.predict(test_features)
        accuracy = accuracy_score(test_positions, pred)
        cm = confusion_matrix(test_positions, pred)
        
        return {
            "model": self.model_name,
            "accuracy": accuracy,
            "confusion_matrix": cm
        }
    
    def plot_confusion_matrix(self, cm):
        """Visualize confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Position")
        plt.ylabel("True Position")
        plt.title(f"Confusion Matrix - {self.model_name}")
        
        output_path = Config.REPORTS_DIR / f"cm_{self.model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()