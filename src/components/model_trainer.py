import os
import sys
sys.path.append('C:\\Projects\\mlproject')
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
import pickle
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models with hyperparameter tuning and select the best one
        """
        try:
            logging.info("Model Training started")

            # Dictionary of models with hyperparameters for tuning
            models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'Ridge': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.1, 1, 10, 100]
                    }
                },
                'Lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': [0.1, 1, 10, 100]
                    }
                },
                'KNeighborsRegressor': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']
                    }
                },
                'DecisionTreeRegressor': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': [5, 10, 15, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'RandomForestRegressor': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, 30],
                        'min_samples_split': [2, 5]
                    }
                },
                'GradientBoostingRegressor': {
                    'model': GradientBoostingRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.5],
                        'max_depth': [3, 5, 7]
                    }
                },
                'SVR': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['linear', 'rbf', 'poly']
                    }
                },
                'XGBRegressor': {
                    'model': XGBRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.5],
                        'max_depth': [3, 5, 7]
                    }
                },
                'CatBoostRegressor': {
                    'model': CatBoostRegressor(verbose=0),
                    'params': {
                        'iterations': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.5],
                        'depth': [4, 6, 8]
                    }
                }
            }

            model_report = {}
            best_model_name = None
            best_model = None
            best_score = -np.inf

            # Train and evaluate each model
            for model_name, model_info in models.items():
                logging.info(f"Training {model_name}...")

                model = model_info['model']
                params = model_info['params']

                # Hyperparameter tuning if params exist
                if params:
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params,
                        n_iter=5,
                        cv=5,
                        n_jobs=-1,
                        random_state=42
                    )
                    random_search.fit(X_train, y_train)
                    best_model_for_this = random_search.best_estimator_
                    logging.info(f"{model_name} Best params: {random_search.best_params_}")
                else:
                    # Train model without tuning
                    best_model_for_this = model
                    best_model_for_this.fit(X_train, y_train)

                # Make predictions
                y_train_pred = best_model_for_this.predict(X_train)
                y_test_pred = best_model_for_this.predict(X_test)

                # Evaluate model
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = np.sqrt(test_mse)

                model_report[model_name] = {
                    'model': best_model_for_this,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'test_mse': test_mse,
                    'test_rmse': test_rmse
                }

                logging.info(f"{model_name} - Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

                # Track best model
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model_name = model_name
                    best_model = best_model_for_this

            # Log model comparison
            logging.info("\n" + "="*70)
            logging.info("MODEL COMPARISON REPORT")
            logging.info("="*70)
            for model_name, metrics in model_report.items():
                logging.info(f"\n{model_name}:")
                logging.info(f"  Train R²: {metrics['train_r2']:.4f}")
                logging.info(f"  Test R²:  {metrics['test_r2']:.4f}")
                logging.info(f"  Test MAE: {metrics['test_mae']:.4f}")
                logging.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")

            logging.info("\n" + "="*70)
            logging.info(f"BEST MODEL: {best_model_name} with Test R² = {best_score:.4f}")
            logging.info("="*70)

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            # Save best model
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)

            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            return {
                'best_model_name': best_model_name,
                'best_model': best_model,
                'model_report': model_report,
                'best_model_path': self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation

    # Initiate data ingestion
    logging.info("Starting data ingestion...")
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Initiate data transformation
    logging.info("Starting data transformation...")
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

    # Initiate model training
    logging.info("Starting model training...")
    model_trainer = ModelTrainer()
    result = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETED")
    print("="*70)
    print(f"\nBest Model: {result['best_model_name']}")
    print(f"Model saved at: {result['best_model_path']}")
    print("\nModel Performance Summary:")
    for model_name, metrics in result['model_report'].items():
        print(f"\n{model_name}:")
        print(f"  Test R²:  {metrics['test_r2']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print("="*70)
