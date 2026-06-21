import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_file_path: str = os.path.join('artifacts', 'model_report.txt')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models and select the best one
        """
        try:
            logging.info("Starting model training")
            
            # Split arrays into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define models to train
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Support Vector Machine": SVC(kernel='rbf', random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
            }
            
            best_model_name = None
            best_model = None
            best_accuracy = 0
            best_recall = 0
            best_classification_report = None
            best_confusion_matrix = None
            
            # Train and evaluate each model
            for name, model in models.items():
                logging.info(f"Training {name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='weighted')
                
                logging.info(f"{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    best_model = model
                    best_recall = recall
                    best_confusion_matrix = confusion_matrix(y_test, y_pred)
                    best_classification_report = classification_report(y_test, y_pred)
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Save model report
            self._save_model_report(
                best_model_name,
                best_accuracy,
                best_recall,
                best_confusion_matrix,
                best_classification_report
            )
            
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Accuracy: {best_accuracy:.4f}")
            
            return best_model_name, best_accuracy, best_model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_model_report(self, name, accuracy, recall, confusion_matrix, classification_report):
        """Save model evaluation report"""
        try:
            with open(self.model_trainer_config.model_report_file_path, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("MODEL EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Best Model: {name}\n")
                f.write(f"Best Accuracy: {accuracy:.4f}\n")
                f.write(f"Best Recall: {recall:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(confusion_matrix) + "\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report)
                f.write("\n" + "=" * 50 + "\n")
            
            logging.info(f"Model report saved to {self.model_trainer_config.model_report_file_path}")
            
        except Exception as e:
            logging.error(f"Error saving model report: {str(e)}")
    
    def cross_validate_best_model(self, X, y, cv=5):
        """Perform cross-validation on the best model"""
        try:
            model = load_object(self.model_trainer_config.trained_model_file_path)
            scores = cross_val_score(model, X, y, cv=cv)
            
            logging.info(f"Cross-validation scores: {scores}")
            logging.info(f"Mean CV score: {scores.mean():.4f}")
            
            return scores.mean()
            
        except Exception as e:
            raise CustomException(e, sys)