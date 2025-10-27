"""
Predictions Module
Real-time prediction functions for the dashboard
LO4: 4.2 - ML Model Predictions
"""

import shap


def make_predictions(input_data, churn_model, ltv_model, cluster_model):
    """
    Make predictions for churn, LTV, and cluster assignment

    Args:
        input_data (pd.DataFrame): Customer input data
        churn_model: Trained churn classifier
        ltv_model: Trained LTV regressor
        cluster_model: Trained clustering model

    Returns:
        dict: Prediction results with explanations
    """
    results = {}

    # Preprocess input data (assume it's already encoded)
    # This would need to match the preprocessing used during training

    try:
        # Churn prediction
        if churn_model is not None:
            churn_proba = churn_model.predict_proba(input_data)[0]
            results['churn_prob'] = churn_proba[1]
            results['churn_prediction'] = (
                'Yes' if churn_proba[1] > 0.5 else 'No'
            )

            # SHAP explanation for churn
            try:
                explainer = shap.TreeExplainer(churn_model)
                shap_values = explainer.shap_values(input_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                results['shap_values'] = shap_values[0]
                results['shap_features'] = input_data.columns.tolist()
            except BaseException:
                results['shap_values'] = None
        else:
            results['churn_prob'] = 0.0
            results['churn_prediction'] = 'Model not available'

        # LTV prediction
        if ltv_model is not None:
            ltv_pred = ltv_model.predict(input_data)[0]
            results['ltv'] = max(0, ltv_pred)  # Ensure non-negative
        else:
            results['ltv'] = 0.0

        # Cluster assignment
        if cluster_model is not None:
            cluster_id = cluster_model.predict(input_data)[0]
            results['cluster'] = int(cluster_id)
            results['cluster_name'] = get_cluster_name(cluster_id)
        else:
            results['cluster'] = 0
            results['cluster_name'] = 'Unknown'

        # Retention recommendation
        results['recommendation'] = get_retention_recommendation(
            results.get('churn_prob', 0),
            results.get('ltv', 0),
            results.get('cluster', 0)
        )

        # ROI estimate
        results['roi'] = calculate_roi(
            results.get('churn_prob', 0),
            results.get('ltv', 0)
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        results = {
            'error': str(e),
            'churn_prob': 0.0,
            'ltv': 0.0,
            'cluster': 0,
            'recommendation': 'Error in prediction'
        }

    return results


def get_cluster_name(cluster_id):
    """
    Get descriptive name for cluster

    Args:
        cluster_id (int): Cluster ID

    Returns:
        str: Cluster description
    """
    cluster_profiles = {
        0: "Young Tech-Savvy (Month-to-Month)",
        1: "Senior Basic Service (High Risk)",
        2: "Family Bundle (Medium Risk)",
        3: "Long-term Loyal (Low Risk)",
        4: "Premium Fiber Users (Medium Risk)",
        5: "Budget Conscious (High Risk)",
        6: "New Customers (Medium Risk)",
        7: "Corporate Contracts (Low Risk)",
        8: "Streaming Enthusiasts (Medium Risk)",
        9: "Basic Phone Only (High Risk)"
    }
    return cluster_profiles.get(cluster_id, f"Cluster {cluster_id}")


def get_retention_recommendation(churn_prob, ltv, cluster_id):
    """
    Generate retention recommendation based on predictions

    Args:
        churn_prob (float): Churn probability (0-1)
        ltv (float): Lifetime value in USD
        cluster_id (int): Cluster ID (0-9)

    Returns:
        str: Retention recommendation with priority level
    """
    if churn_prob > 0.7:
        if ltv > 2000:
            return ("🔴 HIGH PRIORITY: Immediate personal contact + "
                    "20% discount offer")
        elif ltv > 1000:
            return "🟠 MEDIUM PRIORITY: Call within 48h + service upgrade offer"
        else:
            return "🟡 LOW PRIORITY: Automated email campaign + basic discount"
    elif churn_prob > 0.4:
        if ltv > 2000:
            return "🟠 WATCH LIST: Proactive engagement + loyalty rewards"
        else:
            return "🟢 MONITOR: Standard retention campaign"
    else:
        return "🟢 LOW RISK: Continue standard service + upsell opportunities"


def calculate_roi(churn_prob, ltv):
    """
    Calculate estimated ROI for retention intervention

    Args:
        churn_prob (float): Churn probability
        ltv (float): Customer lifetime value

    Returns:
        float: Estimated ROI in USD
    """
    intervention_cost = 50  # Average cost of retention intervention
    success_rate = 0.3  # 30% success rate for interventions

    expected_value = (churn_prob * ltv * success_rate) - intervention_cost
    roi = max(0, expected_value)
    
    # Add ROI category for better interpretation
    if roi > 100:
        roi_category = "High Priority"
    elif roi > 50:
        roi_category = "Medium Priority"
    elif roi > 0:
        roi_category = "Low Priority"
    else:
        roi_category = "Skip"
    
    return roi, roi_category


def predict_churn_batch(df, churn_model, preprocessor):
    """
    Batch churn predictions for entire dataset

    Args:
        df (pd.DataFrame): Customer dataset
        churn_model: Trained churn model
        preprocessor: Preprocessing pipeline

    Returns:
        pd.DataFrame: Dataset with predictions
    """
    from src.preprocessing import encode_input_data

    df_pred = df.copy()

    # Prepare features
    exclude_cols = ['customerID', 'Churn', 'TotalCharges', 'LTV']
    X = df_pred.drop(
        [col for col in exclude_cols if col in df_pred.columns], axis=1)

    # Encode
    X_encoded = encode_input_data(X, preprocessor['label_encoders'])

    # Predict
    churn_proba = churn_model.predict_proba(X_encoded)[:, 1]
    df_pred['ChurnProbability'] = churn_proba
    df_pred['PredictedChurn'] = (churn_proba > 0.5).astype(int)

    return df_pred
