"""
Data Loading Module
Handles data loading and initial processing for the Customer Loyalty
and LTV Predictor
LO7: 7.1 - Data Collection
"""

import pandas as pd
import os


def load_data():
    """
    Load telecom churn dataset (LO7: 7.1)

    Returns:
        pd.DataFrame: Cleaned customer dataset
    """
    try:
        file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {len(df)} records")

            # Initial cleaning (LO7: 7.2)
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce')
            df = df.dropna(subset=['TotalCharges'])

            # Calculate LTV for non-churners
            df['LTV'] = df['MonthlyCharges'] * df['tenure']
            print(f"After cleaning: {len(df)} records")

            return df
        else:
            print(f"File not found: {file_path}")
            print("Please download the dataset from Kaggle and place it in "
                  "the data/ folder")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_ltv(df):
    """
    Calcola Customer Lifetime Value
    LTV = MonthlyCharges * tenure per non-churners

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        pd.DataFrame: Dataset with LTV values for non-churned customers
    """
    df_ltv = df.copy()
    df_ltv['LTV'] = df_ltv['MonthlyCharges'] * df_ltv['tenure']
    return df_ltv[df_ltv['Churn'] == 'No']  # Solo per non-churners


def get_feature_columns(df):
    """
    Get feature columns for ML models

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        list: List of feature column names
    """
    exclude_cols = ['customerID', 'Churn', 'TotalCharges', 'LTV']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def get_categorical_features(df):
    """
    Get categorical feature columns

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        list: List of categorical feature names
    """
    categorical_features = df.select_dtypes(
        include=['object']).columns.tolist()
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    if 'Churn' in categorical_features:
        categorical_features.remove('Churn')
    return categorical_features


def get_numerical_features(df):
    """
    Get numerical feature columns

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        list: List of numerical feature names
    """
    numerical_features = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    exclude = ['customerID', 'TotalCharges', 'LTV']
    numerical_features = [
        col for col in numerical_features if col not in exclude]
    return numerical_features
