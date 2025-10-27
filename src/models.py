"""
Models Module
ML model training and evaluation functions
LO5: 5.1, 5.2 - Model Training and Evaluation
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, r2_score, mean_absolute_error, mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib


def train_churn_model(X_train, y_train, hyperparameter_tuning=True):
    """
    Train churn prediction model (Binary Classification)
    LO5: 5.1 - Model Training

    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform grid search

    Returns:
        model: Trained XGBoost classifier
    """
    if hyperparameter_tuning:
        # GridSearchCV with 6+ hyperparameters (Merit LO5: 5.7)
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best recall score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_
    else:
        # Default parameters
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model


def train_ltv_model(X_train, y_train, hyperparameter_tuning=True):
    """
    Train LTV prediction model (Regression)
    LO5: 5.1 - Model Training

    Args:
        X_train: Training features
        y_train: Training target (LTV values)
        hyperparameter_tuning: Whether to perform grid search

    Returns:
        model: Trained XGBoost regressor
    """
    if hyperparameter_tuning:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 1.0]
        }

        xgb = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best R² score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_
    else:
        model = XGBRegressor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model


def train_clustering_model(X_scaled, n_clusters=7):
    """
    Train customer segmentation model (Clustering)
    LO5: 5.1 - Model Training

    Args:
        X_scaled: Scaled features
        n_clusters: Number of clusters (5-10)

    Returns:
        model: Trained KMeans model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)

    silhouette = silhouette_score(X_scaled, model.labels_)
    print(f"Silhouette Score: {silhouette:.4f}")

    return model


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate classification model (LO5: 5.2)

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    return metrics


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model (LO5: 5.2)

    Args:
        model: Trained regressor
        X_test: Test features
        y_test: Test target

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

    return metrics


def save_model(model, filepath):
    """
    Save trained model to file

    Args:
        model: Trained model
        filepath: Path to save model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_models():
    """
    Load all trained models with error handling

    Returns:
        tuple: churn_model, ltv_model, cluster_model
    """
    try:
        churn_model = joblib.load('models/churn_model.pkl')
        ltv_model = joblib.load('models/ltv_model.pkl')
        cluster_model = joblib.load('models/cluster_model.pkl')
        print("All models loaded successfully!")
        return churn_model, ltv_model, cluster_model
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None
