import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X):
        """
        This function is responsible for creating and returning a ColumnTransformer
        with OneHotEncoder for categorical features and StandardScaler for numerical features
        """
        try:
            # Separate numerical and categorical features
            num_features = X.select_dtypes(include=['int64', 'float64']).columns
            cat_features = X.select_dtypes(include=['object']).columns

            logging.info(f"Numerical columns: {list(num_features)}")
            logging.info(f"Categorical columns: {list(cat_features)}")

            # Create transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('OneHotEncoder', oh_transformer, cat_features),
                    ('StandardScaler', numeric_transformer, num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_column='math score'):
        """
        Read train and test data, apply transformations, and save preprocessor
        """
        try:
            logging.info("Data Transformation started")

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Separate features and target
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            # Get preprocessor object
            preprocessor = self.get_data_transformer_object(X_train)

            # Fit and transform train data
            X_train_transformed = preprocessor.fit_transform(X_train)
            # Transform test data (fit only on train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            # Save preprocessor object
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor object saved")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion

    # Initiate data ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Initiate data transformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")