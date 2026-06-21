import os
import sys
import joblib
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Save object using joblib"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj, file_path)
        logging.info(f"Object saved at: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load object using joblib"""
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from: {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)

def save_model_report(file_path, model_name, accuracy, recall, confusion_matrix, classification_report):
    """Save model evaluation report"""
    try:
        with open(file_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Model: {model_name}\n")
            f.write(f"Best Accuracy: {accuracy:.4f}\n")
            f.write(f"Best Recall: {recall:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix) + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report)
            f.write("\n" + "=" * 50 + "\n")
        
        logging.info(f"📄 Model report saved at: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)