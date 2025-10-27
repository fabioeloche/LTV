"""
Preprocessing Module
Data preprocessing and feature engineering for ML models
LO7: 7.2 - Data Cleaning and Feature Engineering
"""

# import pandas as pd  # Unused import removed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df, target_column='Churn', test_size=0.2, random_state=42):
    """
    Preprocess data for ML models with encoding and splitting

    Args:
        df (pd.DataFrame): Raw customer dataset
        target_column (str): Target column name (default: 'Churn')
        test_size (float): Test set size (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    df_processed = df.copy()

    # Remove customerID
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)

    # Encode target variable
    if target_column in df_processed.columns:
        le = LabelEncoder()
        y = le.fit_transform(df_processed[target_column])
        X = df_processed.drop(target_column, axis=1)
    else:
        raise ValueError(f"Target column '{target_column}' not found")

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state,
        stratify=y)

    preprocessor = {
        'label_encoders': label_encoders,
        'target_encoder': le,
        'feature_names': X_encoded.columns.tolist()
    }

    return X_train, X_test, y_train, y_test, preprocessor


def preprocess_for_clustering(df):
    """
    Preprocess data for clustering (exclude target and IDs)

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        tuple: X_scaled, scaler, feature_names
    """
    df_processed = df.copy()

    # Remove non-feature columns
    exclude_cols = ['customerID', 'Churn', 'TotalCharges', 'LTV']
    for col in exclude_cols:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)

    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed)

    preprocessor = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': df_processed.columns.tolist()
    }

    return X_scaled, preprocessor


def preprocess_for_ltv(df):
    """
    Preprocess data for LTV prediction (non-churned customers only)

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Filter non-churned customers
    df_ltv = df[df['Churn'] == 'No'].copy()

    # Calculate LTV target
    df_ltv['LTV'] = df_ltv['MonthlyCharges'] * df_ltv['tenure']

    # Remove unnecessary columns
    exclude_cols = ['customerID', 'Churn', 'TotalCharges', 'tenure']
    X = df_ltv.drop(
        [col for col in exclude_cols if col in df_ltv.columns] + ['LTV'],
        axis=1)
    y = df_ltv['LTV']

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    preprocessor = {
        'label_encoders': label_encoders,
        'feature_names': X_encoded.columns.tolist()
    }

    return X_train, X_test, y_train, y_test, preprocessor


def encode_input_data(input_df, label_encoders):
    """
    Encode new input data using fitted label encoders

    Args:
        input_df (pd.DataFrame): Input data to encode
        label_encoders (dict): Dictionary of fitted LabelEncoders

    Returns:
        pd.DataFrame: Encoded input data
    """
    input_encoded = input_df.copy()

    for col, encoder in label_encoders.items():
        if col in input_encoded.columns:
            try:
                input_encoded[col] = encoder.transform(input_encoded[col])
            except ValueError as e:
                # Handle unseen categories with warning
                print(f"Warning: Unseen category in {col}: {e}")
                input_encoded[col] = 0

    return input_encoded
