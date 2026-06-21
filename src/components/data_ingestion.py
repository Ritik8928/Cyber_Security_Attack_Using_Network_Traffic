import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def get_dataset_path(self):
        """Find the dataset path regardless of where the script is run from"""
        # Try multiple possible locations
        possible_paths = [
            Path('dataset/cyber_attack_dataset.csv'),  # From project root
            Path('../dataset/cyber_attack_dataset.csv'),  # From src/components
            Path('../../dataset/cyber_attack_dataset.csv'),  # From deeper levels
            Path(__file__).parent.parent.parent / 'dataset' / 'cyber_attack_dataset.csv',  # Absolute from script
            Path.cwd() / 'dataset' / 'cyber_attack_dataset.csv',  # Current working directory
        ]
        
        for path in possible_paths:
            if path.exists():
                logging.info(f"Found dataset at: {path}")
                return path
        
        # If not found, try to create the directory structure
        dataset_dir = Path.cwd() / 'dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if it exists in parent directory (one level up)
        parent_dataset = Path.cwd().parent / 'dataset' / 'cyber_attack_dataset.csv'
        if parent_dataset.exists():
            return parent_dataset
        
        logging.error("   Dataset not found in any expected location")
        logging.error(f"   Searched in: {[str(p) for p in possible_paths]}")
        logging.error("   Please make sure the dataset exists at: dataset/cyber_attack_dataset.csv")
        
        return None
    
    def initiate_data_ingestion(self):
        """
        Reads the dataset and splits it into train and test sets
        """
        logging.info("Entered data ingestion method")
        try:
            # Get the dataset path
            dataset_path = self.get_dataset_path()
            
            if dataset_path is None:
                raise FileNotFoundError("Dataset file not found. Please ensure 'dataset/cyber_attack_dataset.csv' exists.")
            
            # Read the dataset
            df = pd.read_csv(dataset_path)
            logging.info(f"Dataset loaded with shape: {df.shape}")
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Split the data
            train_set, test_set = train_test_split(
                df, 
                test_size=0.20, 
                random_state=42
            )
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

def main():
    """Main function to run data ingestion"""
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(f"\n✅ Data ingestion completed successfully!")
        print(f"   Train data: {train_data}")
        print(f"   Test data: {test_data}")
    except Exception as e:
        print(f"\n❌ Data ingestion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()