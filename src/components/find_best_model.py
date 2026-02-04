import os
import sys
sys.path.append('C:\\Projects\\mlproject')
import pandas as pd
import numpy as np
from src.logger import logging
import pickle

def analyze_model_report(model_report):
    """
    Analyze model report and find best models based on different metrics
    """
    
    # Create a comparison dataframe
    comparison_data = []
    
    for model_name, metrics in model_report.items():
        comparison_data.append({
            'Model': model_name,
            'Train R²': metrics['train_r2'],
            'Test R²': metrics['test_r2'],
            'Train MAE': metrics['train_mae'],
            'Test MAE': metrics['test_mae'],
            'Test MSE': metrics['test_mse'],
            'Test RMSE': metrics['test_rmse']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display full comparison
    print("\n" + "="*100)
    print("COMPLETE MODEL COMPARISON REPORT")
    print("="*100)
    print(df_comparison.to_string(index=False))
    print("="*100)
    
    # Find best models based on different metrics
    print("\n" + "="*100)
    print("BEST MODELS BY DIFFERENT METRICS")
    print("="*100)
    
    best_test_r2 = df_comparison.loc[df_comparison['Test R²'].idxmax()]
    print(f"\n✓ Best by Test R² Score:")
    print(f"  Model: {best_test_r2['Model']}")
    print(f"  Test R²: {best_test_r2['Test R²']:.4f}")
    print(f"  Test MAE: {best_test_r2['Test MAE']:.4f}")
    print(f"  Test RMSE: {best_test_r2['Test RMSE']:.4f}")
    
    best_test_mae = df_comparison.loc[df_comparison['Test MAE'].idxmin()]
    print(f"\n✓ Best by Test MAE (Lower is Better):")
    print(f"  Model: {best_test_mae['Model']}")
    print(f"  Test MAE: {best_test_mae['Test MAE']:.4f}")
    print(f"  Test R²: {best_test_mae['Test R²']:.4f}")
    print(f"  Test RMSE: {best_test_mae['Test RMSE']:.4f}")
    
    best_test_rmse = df_comparison.loc[df_comparison['Test RMSE'].idxmin()]
    print(f"\n✓ Best by Test RMSE (Lower is Better):")
    print(f"  Model: {best_test_rmse['Model']}")
    print(f"  Test RMSE: {best_test_rmse['Test RMSE']:.4f}")
    print(f"  Test R²: {best_test_rmse['Test R²']:.4f}")
    print(f"  Test MAE: {best_test_rmse['Test MAE']:.4f}")
    
    # Check for overfitting
    print("\n" + "="*100)
    print("OVERFITTING ANALYSIS (Train R² - Test R²)")
    print("="*100)
    
    df_comparison['Overfitting'] = df_comparison['Train R²'] - df_comparison['Test R²']
    df_sorted = df_comparison.sort_values('Overfitting')
    
    print("\nModels with Least Overfitting (Best Generalization):")
    print(df_sorted[['Model', 'Train R²', 'Test R²', 'Overfitting']].to_string(index=False))
    
    best_generalization = df_sorted.iloc[0]
    print(f"\n✓ Best Generalization (Least Overfitting):")
    print(f"  Model: {best_generalization['Model']}")
    print(f"  Overfitting Gap: {best_generalization['Overfitting']:.4f}")
    print(f"  Train R²: {best_generalization['Train R²']:.4f}")
    print(f"  Test R²: {best_generalization['Test R²']:.4f}")
    
    # Top 3 models overall
    print("\n" + "="*100)
    print("TOP 3 MODELS (By Test R²)")
    print("="*100)
    top_3 = df_comparison.nlargest(3, 'Test R²')
    for idx, (i, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{idx}. {row['Model']}")
        print(f"   Test R²: {row['Test R²']:.4f}")
        print(f"   Test MAE: {row['Test MAE']:.4f}")
        print(f"   Test RMSE: {row['Test RMSE']:.4f}")
        print(f"   Overfitting Gap: {row['Overfitting']:.4f}")
    
    print("\n" + "="*100)
    
    return {
        'comparison_df': df_comparison,
        'best_by_r2': best_test_r2['Model'],
        'best_by_mae': best_test_mae['Model'],
        'best_by_rmse': best_test_rmse['Model'],
        'best_generalization': best_generalization['Model'],
        'top_3_models': top_3['Model'].tolist()
    }

def get_best_model_info():
    """
    Load the best model and provide information about it
    """
    model_path = 'artifacts/model.pkl'
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    print("\n" + "="*100)
    print("BEST MODEL INFORMATION")
    print("="*100)
    print(f"Model Type: {type(best_model).__name__}")
    print(f"Model Path: {model_path}")
    
    if hasattr(best_model, 'get_params'):
        print("\nModel Hyperparameters:")
        params = best_model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    print("="*100)
    
    return best_model

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

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

    # Analyze model report
    analysis_result = analyze_model_report(result['model_report'])
    
    # Get best model info
    best_model = get_best_model_info()
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"✓ Best Model (Overall): {result['best_model_name']}")
    print(f"✓ Best by R² Score: {analysis_result['best_by_r2']}")
    print(f"✓ Best by MAE: {analysis_result['best_by_mae']}")
    print(f"✓ Best by RMSE: {analysis_result['best_by_rmse']}")
    print(f"✓ Best Generalization: {analysis_result['best_generalization']}")
    print(f"✓ Top 3 Models: {', '.join(analysis_result['top_3_models'])}")
    print("="*100)
