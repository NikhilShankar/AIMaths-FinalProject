import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
import seaborn as sns

class DecisionTreeTrainer:
    #The default argument is for creating a decision tree that goes as deep as needed. We can tweak this while creating
    #the class to avoid overfitting and for faster training times.
    def __init__(self, model_type, result_path, max_depth=None, max_features=None, min_samples_split=2, estimators=50):
        self.model_type = model_type
        self.result_path = result_path
        self.model = None
        self.num_estimators = estimators
        self.decision_tree = clf = DecisionTreeClassifier(
            max_depth=max_depth,  # Limit tree depth
            max_features=max_features,  # Randomly select features
            min_samples_split=min_samples_split  # Prevent overfitting
        )

    def _create_result_folder(self):
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        folder_name = f"result-{self.model_type}-{timestamp}"
        result_folder = os.path.join(self.result_path, folder_name)
        os.makedirs(result_folder, exist_ok=True)
        return result_folder
    
    def train(self, x_train_path, y_train_path, x_test_path, y_test_path, train_count=None, test_count=None):
        x_train, y_train, x_test, y_test = self._load_data(x_train_path, y_train_path, x_test_path, y_test_path, train_count=None, test_count=None)
        return self._train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def _load_data(self, x_train_path, y_train_path, x_test_path, y_test_path, train_count=None, test_count=None):
        # Load datasets
        x_train = pd.read_csv(x_train_path).values
        y_train = pd.read_csv(y_train_path).values.flatten()
        x_test = pd.read_csv(x_test_path).values
        y_test = pd.read_csv(y_test_path).values.flatten()

        # Reduce dataset size if parameters are specified
        if train_count:
            x_train, y_train = shuffle(x_train, y_train, random_state=42)
            x_train, y_train = x_train[:train_count], y_train[:train_count]

        if test_count:
            x_test, y_test = shuffle(x_test, y_test, random_state=42)
            x_test, y_test = x_test[:test_count], y_test[:test_count]
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    
    

    def _train(self, x_train, y_train, x_test, y_test):
        result_folder = self._create_result_folder()
        
        # Define model
        if self.model_type == "single":
            self.model = self.decision_tree
        elif self.model_type == "boosting":
            self.model = AdaBoostClassifier(base_estimator=self.decision_tree, n_estimators=self.num_estimators)
        elif self.model_type == "bagging":
            self.model = BaggingClassifier(base_estimator=self.decision_tree, n_estimators=self.num_estimators)
        else:
            raise ValueError("Invalid model type. Choose from 'single', 'boosting', or 'bagging'.")

        # 5-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        fold_metrics = []
        start_train = time.time()
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
            x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            self.model.fit(x_train_fold, y_train_fold)
            y_val_pred = self.model.predict(x_val_fold)

            acc = accuracy_score(y_val_fold, y_val_pred)
            precision = precision_score(y_val_fold, y_val_pred, average="macro")
            recall = recall_score(y_val_fold, y_val_pred, average="macro")
            f1 = f1_score(y_val_fold, y_val_pred, average="macro")

            fold_metrics.append({
                "Fold": fold + 1,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

        train_time = time.time() - start_train

        # Train final model on full training data
        self.model.fit(x_train, y_train)

        # Evaluate on test set
        start_val = time.time()
        y_pred = self.model.predict(x_test)
        val_time = time.time() - start_val

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Save metrics
        metrics_csv_path = os.path.join(result_folder, "metrics.csv")
        fold_metrics_df = pd.DataFrame(fold_metrics)
        summary_metrics = {
            "Fold": "Total",
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Train Time (s)": train_time,
            "Validation Time (s)": val_time
        }
        summary_metrics_df = pd.DataFrame([summary_metrics])  # Create a DataFrame for the summary row
        fold_metrics_df = pd.concat([fold_metrics_df, summary_metrics_df], ignore_index=True)  # Concatenate
        fold_metrics_df.to_csv(metrics_csv_path, index=False)
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Create a figure and plot the confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(result_folder, "confusion_matrix.png"))
        plt.show()
        return fold_metrics_df

    def predict_image(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img).reshape(1, -1)

        # Predict the class
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model first.")

        pred = self.model.predict(img_array)

        # Display the image and prediction
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted Class: {pred[0]}")
        plt.axis("off")
        plt.show()
        return pred[0]


