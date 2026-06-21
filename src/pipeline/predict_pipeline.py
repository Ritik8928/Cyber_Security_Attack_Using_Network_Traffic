import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'
        self.label_encoder_path = 'artifacts/label_encoder.pkl'
    
    def predict(self, features):
        try:
            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            label_encoder = load_object(self.label_encoder_path)
            
            # Preprocess features
            # Handle protocol column
            if 'protocol' in features.columns:
                features['protocol'] = features['protocol'].map({'TCP': 0, 'UDP': 1})
            
            # Transform features
            features_transformed = preprocessor.transform(features)
            
            # Make predictions
            predictions_encoded = model.predict(features_transformed)
            
            # Decode predictions
            predictions = label_encoder.inverse_transform(predictions_encoded)
            
            return predictions
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_proba(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            
            if 'protocol' in features.columns:
                features['protocol'] = features['protocol'].map({'TCP': 0, 'UDP': 1})
            
            features_transformed = preprocessor.transform(features)
            probabilities = model.predict_proba(features_transformed)
            
            return probabilities
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, duration, src_bytes, dst_bytes, packet_count, protocol, failed_logins):
        self.duration = duration
        self.src_bytes = src_bytes
        self.dst_bytes = dst_bytes
        self.packet_count = packet_count
        self.protocol = protocol
        self.failed_logins = failed_logins
    
    def get_data_as_dataframe(self):
        """Convert input data to DataFrame"""
        try:
            custom_data_input_dict = {
                "duration": [self.duration],
                "src_bytes": [self.src_bytes],
                "dst_bytes": [self.dst_bytes],
                "packet_count": [self.packet_count],
                "protocol": [self.protocol],
                "failed_logins": [self.failed_logins]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)