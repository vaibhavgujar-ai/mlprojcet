import os
import sys
sys.path.append('C:\\Projects\\mlproject')
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    """
    Pipeline for making predictions using trained model and preprocessor
    """
    
    def __init__(self):
        """Initialize the prediction pipeline by loading artifacts"""
        self.preprocessor = None
        self.model = None
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        
        # Load artifacts on initialization
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load preprocessor and model from disk"""
        try:
            logging.info("Loading artifacts for prediction...")
            
            # Check if files exist
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load preprocessor
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logging.info("Preprocessor loaded successfully")
            
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading artifacts: {str(e)}")
            raise CustomException(e, sys)
    
    def validate_input(self, data):
        """
        Validate input data
        
        Args:
            data (dict): Dictionary with required columns
            
        Returns:
            bool: True if valid, raises exception otherwise
        """
        required_columns = {
            'gender', 'race/ethnicity', 'parental level of education',
            'lunch', 'test preparation course', 'reading score', 'writing score'
        }
        
        provided_columns = set(data.keys())
        
        if not required_columns.issubset(provided_columns):
            missing = required_columns - provided_columns
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
    
    def predict_single(self, data):
        """
        Make a single prediction
        
        Args:
            data (dict): Dictionary with student features
            
        Returns:
            float: Predicted math score
        """
        try:
            logging.info("Making single prediction...")
            
            # Validate input
            self.validate_input(data)
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(df)
            logging.info(f"Data transformed. Shape: {X_transformed.shape}")
            
            # Make prediction
            prediction = self.model.predict(X_transformed)[0]
            
            # Clamp prediction to valid range (0-100)
            prediction = max(0, min(100, float(prediction)))
            
            logging.info(f"Prediction made: {prediction:.2f}")
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error during single prediction: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_batch(self, data_list):
        """
        Make batch predictions for multiple records
        
        Args:
            data_list (list): List of dictionaries, each with student features
            
        Returns:
            list: List of predicted math scores
        """
        try:
            logging.info(f"Making batch predictions for {len(data_list)} records...")
            
            if not isinstance(data_list, list):
                raise ValueError("Expected list of dictionaries")
            
            if len(data_list) == 0:
                raise ValueError("Empty data list provided")
            
            # Validate first record to check all columns exist
            self.validate_input(data_list[0])
            
            # Create DataFrame from list
            df = pd.DataFrame(data_list)
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(df)
            logging.info(f"Data transformed. Shape: {X_transformed.shape}")
            
            # Make predictions
            predictions = self.model.predict(X_transformed)
            
            # Clamp predictions to valid range (0-100)
            predictions = np.clip(predictions, 0, 100).tolist()
            
            logging.info(f"Batch predictions completed. Count: {len(predictions)}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during batch prediction: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_from_dataframe(self, df):
        """
        Make predictions from a pandas DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with student features
        
        Returns:
            np.ndarray: Array of predicted math scores
        """
        try:
            logging.info(f"Making predictions from DataFrame with shape {df.shape}...")
            
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.model.predict(X_transformed)
            
            # Clamp predictions to valid range (0-100)
            predictions = np.clip(predictions, 0, 100)
            
            logging.info(f"DataFrame predictions completed. Count: {len(predictions)}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during DataFrame prediction: {str(e)}")
            raise CustomException(e, sys)
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'SVR (Support Vector Regressor)',
            'features_expected': 157,
            'preprocessing': 'OneHotEncoder (categorical) + StandardScaler (numerical)',
            'target': 'Math Score (0-100)',
            'r2_score': 0.6536,
            'test_mae': 5.9336,
            'test_rmse': 9.1810,
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PREDICTION PIPELINE - TESTING")
    print("="*70)
    
    try:
        # Initialize pipeline
        predictor = PredictionPipeline()
        print("\n✓ Pipeline initialized successfully")
        
        # Print model info
        print("\nModel Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test single prediction
        print("\n" + "="*70)
        print("SINGLE PREDICTION TEST")
        print("="*70)
        
        test_data = {
            'gender': 'male',
            'race/ethnicity': 'group A',
            'parental level of education': "bachelor's degree",
            'lunch': 'standard',
            'test preparation course': 'completed',
            'reading score': 85,
            'writing score': 88
        }
        
        prediction = predictor.predict_single(test_data)
        print(f"\nInput data: {test_data}")
        print(f"Predicted Math Score: {prediction:.2f}")
        
        # Test batch prediction
        print("\n" + "="*70)
        print("BATCH PREDICTION TEST")
        print("="*70)
        
        test_data_list = [
            {
                'gender': 'male',
                'race/ethnicity': 'group A',
                'parental level of education': "bachelor's degree",
                'lunch': 'standard',
                'test preparation course': 'completed',
                'reading score': 85,
                'writing score': 88
            },
            {
                'gender': 'female',
                'race/ethnicity': 'group B',
                'parental level of education': 'high school',
                'lunch': 'free/reduced',
                'test preparation course': 'none',
                'reading score': 70,
                'writing score': 72
            }
        ]
        
        predictions = predictor.predict_batch(test_data_list)
        print(f"\nBatch predictions (2 records):")
        for i, pred in enumerate(predictions, 1):
            print(f"  Student {i}: {pred:.2f}")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
