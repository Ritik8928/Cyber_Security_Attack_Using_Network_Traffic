import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logging.info("🚀 Starting training pipeline")
            
            # Step 1: Data Ingestion
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            
            # Step 3: Model Training
            best_model_name, best_accuracy, best_model = self.model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )
            
            logging.info(f"Training pipeline completed successfully!")
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Accuracy: {best_accuracy:.4f}")
            
            return {
                'best_model_name': best_model_name,
                'best_accuracy': best_accuracy,
                'best_model': best_model,
                'preprocessor_path': preprocessor_path
            }
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    result = pipeline.run_pipeline()
    print(f"\n{'='*50}")
    print(f"Best Model: {result['best_model_name']}")
    print(f"Best Accuracy: {result['best_accuracy']:.4f}")
    print(f"{'='*50}")