import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from pathlib import Path
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        Creates a preprocessor pipeline with:
        1. Log transformation for skewed features
        2. Robust scaling (handles outliers naturally)
        3. Label encoding for categorical features
        """
        try:
            # Define columns based on your dataset
            numeric_features = [
                'duration', 'src_bytes', 'dst_bytes', 
                'packet_count', 'failed_logins'
            ]
            
            categorical_features = ['protocol']
            
            # Features that are heavily skewed (from your EDA)
            skewed_features = [
                'src_bytes', 'dst_bytes', 
                'packet_count', 'failed_logins'
            ]
            
            # Features that should be scaled normally
            normal_features = ['duration']
            
            # Pipeline for skewed features: Log Transform + Robust Scaling
            skewed_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Median is robust to outliers
                ('log_transform', PowerTransformer(method='yeo-johnson')),  # Handles zeros/negatives
                ('scaler', RobustScaler())  # Uses median/IQR - perfect for outlier-heavy data
            ])
            
            # Pipeline for normal features
            normal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Pipeline for categorical features
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('label_encoder', LabelEncoder())  # Note: LabelEncoder doesn't work well in Pipeline
            ])
            
            # Combine all preprocessors
            preprocessor = ColumnTransformer(
                transformers=[
                    ('skewed', skewed_pipeline, skewed_features),
                    ('normal', normal_pipeline, normal_features),
                ],
                remainder='passthrough'  # Keep other columns as-is
            )
            
            logging.info("Data transformer object created with outlier handling")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Main method to transform train and test data
        
        Args:
            train_path: Path to training CSV
            test_path: Path to testing CSV
        
        Returns:
            train_arr: Transformed training array
            test_arr: Transformed testing array
            preprocessor_path: Saved preprocessor path
        """
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and test data loaded")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            # Handle protocol column manually (as LabelEncoder doesn't work well in Pipeline)
            train_df['protocol'] = train_df['protocol'].map({'TCP': 0, 'UDP': 1})
            test_df['protocol'] = test_df['protocol'].map({'TCP': 0, 'UDP': 1})
            
            # Identify target column
            target_column = 'attack_type'
            if target_column not in train_df.columns:
                common_targets = ['label', 'class', 'target']
                for col in common_targets:
                    if col in train_df.columns:
                        target_column = col
                        break
                else:
                    target_column = train_df.columns[-1]
            
            logging.info(f"Target column: {target_column}")
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Log outlier statistics before transformation
            self._log_outlier_stats(X_train)
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()
            
            # Apply transformation
            logging.info("Applying transformation pipeline...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Encode target labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Combine transformed features with target
            train_arr = np.c_[
                X_train_transformed,
                np.array(y_train_encoded).reshape(-1, 1)
            ]
            
            test_arr = np.c_[
                X_test_transformed,
                np.array(y_test_encoded).reshape(-1, 1)
            ]
            
            logging.info(f"Transformation complete. Train shape: {train_arr.shape}")
            
            # Save preprocessor and label encoder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            # Save label encoder separately
            save_object(
                file_path=os.path.join('artifacts', 'label_encoder.pkl'),
                obj=le
            )
            
            logging.info(f"Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _log_outlier_stats(self, df):
        """Log outlier statistics for each numeric column"""
        try:
            logging.info("===== OUTLIER STATISTICS BEFORE TRANSFORMATION =====")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outliers_iqr = ((df[col] < lower) | (df[col] > upper)).sum()
                pct = (outliers_iqr / len(df)) * 100
                
                logging.info(f" {col}:")
                logging.info(f"   - IQR Outliers: {outliers_iqr} ({pct:.2f}%)")
                logging.info(f"   - 99th Percentile: {df[col].quantile(0.99):.2f}")
                logging.info(f"   - Max value: {df[col].max():.2f}")
                logging.info("   ---")
                
        except Exception as e:
            logging.warning(f"Could not log outlier stats: {str(e)}")