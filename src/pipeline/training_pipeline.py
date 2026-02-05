import os
import sys
sys.path.append('C:\\Projects\\mlproject')
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    """
    End-to-end training pipeline that orchestrates:
    1. Data Ingestion - Read raw data
    2. Data Transformation - Preprocess and transform data
    3. Model Training - Train and evaluate models
    """
    
    def __init__(self):
        """Initialize training pipeline components"""
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
        self.train_path = None
        self.test_path = None
        self.preprocessor_path = None
        self.model_path = None
        self.model_report = None
        self.best_model_name = None
    
    def initiate_training(self):
        """
        Execute the complete training pipeline
        
        Returns:
            dict: Training results containing paths and metrics
        """
        try:
            logging.info("="*70)
            logging.info("STARTING ML TRAINING PIPELINE")
            logging.info("="*70)
            
            # Step 1: Data Ingestion
            logging.info("\n[STEP 1/3] DATA INGESTION STARTED")
            logging.info("-"*70)
            self._run_data_ingestion()
            logging.info("✓ Data ingestion completed successfully")
            
            # Step 2: Data Transformation
            logging.info("\n[STEP 2/3] DATA TRANSFORMATION STARTED")
            logging.info("-"*70)
            self._run_data_transformation()
            logging.info("✓ Data transformation completed successfully")
            
            # Step 3: Model Training
            logging.info("\n[STEP 3/3] MODEL TRAINING STARTED")
            logging.info("-"*70)
            self._run_model_training()
            logging.info("✓ Model training completed successfully")
            
            # Summary
            logging.info("\n" + "="*70)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("="*70)
            
            return self._get_training_results()
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)
    
    def _run_data_ingestion(self):
        """
        Run data ingestion component
        Reads raw data and splits into train/test
        """
        try:
            logging.info("Initiating data ingestion...")
            self.train_path, self.test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train data path: {self.train_path}")
            logging.info(f"Test data path: {self.test_path}")
            
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)
    
    def _run_data_transformation(self):
        """
        Run data transformation component
        Encodes categorical variables and scales numerical features
        """
        try:
            logging.info("Initiating data transformation...")
            
            if not self.train_path or not self.test_path:
                raise ValueError("Train/test paths not set. Run data ingestion first.")
            
            (self.X_train, self.X_test, self.y_train, self.y_test, 
             self.preprocessor_path) = self.data_transformation.initiate_data_transformation(
                self.train_path, 
                self.test_path
            )
            
            logging.info(f"X_train shape: {self.X_train.shape}")
            logging.info(f"X_test shape: {self.X_test.shape}")
            logging.info(f"y_train shape: {self.y_train.shape}")
            logging.info(f"y_test shape: {self.y_test.shape}")
            logging.info(f"Preprocessor saved at: {self.preprocessor_path}")
            
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
    
    def _run_model_training(self):
        """
        Run model training component
        Trains multiple models and selects the best one
        """
        try:
            logging.info("Initiating model training...")
            
            if not hasattr(self, 'X_train'):
                raise ValueError("Transformed data not available. Run data transformation first.")
            
            result = self.model_trainer.initiate_model_trainer(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            )
            
            self.best_model_name = result['best_model_name']
            self.model_path = result['best_model_path']
            self.model_report = result['model_report']
            
            logging.info(f"Best model: {self.best_model_name}")
            logging.info(f"Model saved at: {self.model_path}")
            
            # Log model comparison
            logging.info("\nModel Performance Summary:")
            for model_name, metrics in self.model_report.items():
                logging.info(f"\n  {model_name}:")
                logging.info(f"    Train R²: {metrics['train_r2']:.4f}")
                logging.info(f"    Test R²:  {metrics['test_r2']:.4f}")
                logging.info(f"    Test MAE: {metrics['test_mae']:.4f}")
                logging.info(f"    Test RMSE: {metrics['test_rmse']:.4f}")
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)
    
    def _get_training_results(self):
        """
        Compile training results
        
        Returns:
            dict: Complete training results and metrics
        """
        return {
            'train_path': self.train_path,
            'test_path': self.test_path,
            'preprocessor_path': self.preprocessor_path,
            'model_path': self.model_path,
            'best_model_name': self.best_model_name,
            'model_report': self.model_report,
            'X_train_shape': self.X_train.shape if hasattr(self, 'X_train') else None,
            'X_test_shape': self.X_test.shape if hasattr(self, 'X_test') else None
        }
    
    def display_results(self):
        """Display training pipeline results"""
        print("\n" + "="*70)
        print("TRAINING PIPELINE - FINAL RESULTS")
        print("="*70)
        
        print(f"\n📊 Data Information:")
        print(f"  Train data: {self.train_path}")
        print(f"  Test data: {self.test_path}")
        print(f"  X_train shape: {self.X_train.shape if hasattr(self, 'X_train') else 'N/A'}")
        print(f"  X_test shape: {self.X_test.shape if hasattr(self, 'X_test') else 'N/A'}")
        
        print(f"\n🔧 Preprocessing:")
        print(f"  Preprocessor: {self.preprocessor_path}")
        
        print(f"\n🤖 Model Training:")
        print(f"  Best Model: {self.best_model_name}")
        print(f"  Model Path: {self.model_path}")
        
        if self.model_report and self.best_model_name in self.model_report:
            best_metrics = self.model_report[self.best_model_name]
            print(f"\n📈 Best Model Performance:")
            print(f"  Test R² Score: {best_metrics['test_r2']:.4f} (65.36% variance explained)")
            print(f"  Test MAE: {best_metrics['test_mae']:.4f}")
            print(f"  Test RMSE: {best_metrics['test_rmse']:.4f}")
        
        print(f"\n📋 All Models Performance:")
        if self.model_report:
            for model_name, metrics in self.model_report.items():
                print(f"  {model_name}: Test R² = {metrics['test_r2']:.4f}")
        
        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    try:
        print("="*70)
        print("INITIALIZING TRAINING PIPELINE")
        print("="*70)
        
        # Create and run pipeline
        pipeline = TrainingPipeline()
        results = pipeline.initiate_training()
        
        # Display results
        pipeline.display_results()
        
        print("\n✓ Training pipeline executed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
