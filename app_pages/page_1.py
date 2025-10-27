"""
Page 1: Project Overview
LO6: 6.1 - Dashboard Page Design
LO3: 3.1 - Dataset Description
"""

import streamlit as st
import pandas as pd

def render(df):
    """
    Render Project Overview page
    
    Args:
        df (pd.DataFrame): Customer dataset
    """
    st.title("Customer Loyalty & LTV Predictor")
    st.markdown("### Portfolio Project 5: Predictive Analytics")
    
    st.markdown("---")
    
    # Project Summary
    st.header("Project Goals")
    st.markdown("""
    The **Customer Loyalty and LTV Predictor** is a machine learning application designed for telecom companies to:
    
    - **Analyze loyalty patterns** to identify key churn drivers
    - **Predict churn risk** for customers/prospects with explainability
    - **Segment customers** into micro-clusters for targeted marketing
    - **Recommend retention actions** with ROI analysis
    - **Predict Customer Lifetime Value** to prioritize high-value customers
    """)
    
    # Target Audience
    st.header("Target Audience")
    st.markdown("""
    This application is designed for:
    - **Sales Teams**: Identify at-risk customers for proactive interventions
    - **Marketing Managers**: Segment customers for targeted retention campaigns
    - **Business Analysts**: Understand churn patterns and optimize strategies
    """)
    
    st.markdown("---")
    
    # Dataset Description (LO3: 3.1)
    st.header("Dataset Description")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        if 'Churn' in df.columns:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("""
    **Source**: Telecom Churn Dataset from Kaggle (~7,043 records)
    
    **Features Categories**:
    - **Demographics**: gender, SeniorCitizen, Partner, Dependents
    - **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, etc.
    - **Contract/Billing**: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    - **Target**: Churn (Yes/No)
    """)
    
    # Display sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
    # Dataset statistics
    with st.expander("Dataset Statistics"):
        st.write(df.describe())
    
    st.markdown("---")
    
    # Business Requirements
    st.header("Business Requirements")
    
    requirements = [
        {
            "id": 1,
            "title": "Analyze Loyalty Patterns",
            "description": "Identify key variables correlated with churn and loyalty through EDA",
            "page": "Loyalty Pattern Analysis"
        },
        {
            "id": 2,
            "title": "Predict Churn Risk",
            "description": "Forecast churn probability with explainability (SHAP) for customers/prospects",
            "page": "Loyalty & LTV Predictor"
        },
        {
            "id": 3,
            "title": "Segment Customers",
            "description": "Create micro-clusters (5-10) for targeted retention strategies",
            "page": "Loyalty & LTV Predictor"
        },
        {
            "id": 4,
            "title": "Recommend Retention Actions",
            "description": "Suggest personalized retention strategies with ROI analysis",
            "page": "Retention ROI Analysis"
        },
        {
            "id": 5,
            "title": "Predict Customer Lifetime Value",
            "description": "Estimate LTV to prioritize high-value customers",
            "page": "Loyalty & LTV Predictor"
        }
    ]
    
    for req in requirements:
        with st.expander(f"**Requirement {req['id']}**: {req['title']}"):
            st.markdown(f"**Description**: {req['description']}")
            st.markdown(f"**Dashboard Page**: {req['page']}")
    
    st.markdown("---")
    
    # Hypotheses (Merit LO1: 1.2)
    st.header("Project Hypotheses")
    
    st.markdown("""
    ### Hypothesis 1: Long-term contracts reduce churn
    - **Validation**: Spearman correlation (r = -0.45) and barplot visualization
    - **Status**: ✅ Validated
    
    ### Hypothesis 2: Additional services increase loyalty
    - **Validation**: Feature importance analysis shows services are top predictors
    - **Status**: ✅ Validated
    
    ### Hypothesis 3: High monthly charges correlate with higher LTV if retained
    - **Validation**: Statistical t-test (p < 0.05) and regression analysis
    - **Status**: ✅ Validated
    
    See **Hypothesis Validation** page for detailed analysis.
    """)
    
    st.markdown("---")
    
    # ML Business Case (LO3: 3.2)
    st.header("🤖 ML Business Case")
    
    st.markdown("""
    ### 1. Churn Prediction (Binary Classification)
    - **Objective**: Predict if a customer will churn
    - **Model**: XGBoost Classifier with SHAP explainability
    - **Success Metrics**: Recall ≥80% for Churn class, Precision ≥80% for No Churn
    - **Business Impact**: Enable proactive retention for at-risk customers
    
    ### 2. LTV Prediction (Regression)
    - **Objective**: Estimate Customer Lifetime Value
    - **Model**: XGBoost Regressor
    - **Success Metrics**: R² ≥0.7, MAE <20%
    - **Business Impact**: Prioritize high-value customers for retention
    
    ### 3. Customer Segmentation (Clustering)
    - **Objective**: Group customers into micro-clusters
    - **Model**: K-Means (5-10 clusters)
    - **Success Metrics**: Silhouette Score ≥0.45
    - **Business Impact**: Enable targeted marketing campaigns
    """)
    
    st.markdown("---")
    
    # Technologies
    st.header("Technologies Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Core Technologies**:
        - Python 3.12.1
        - Streamlit (Dashboard)
        - Pandas & NumPy (Data Analysis)
        - Jupyter Notebooks (ML Development)
        """)
    
    with col2:
        st.markdown("""
        **ML & Visualization**:
        - XGBoost (ML Models)
        - SHAP (Explainability)
        - Plotly (Interactive Visualizations)
        - scikit-learn (Preprocessing & Clustering)
        """)
    
    st.markdown("---")
    
    # Next Steps
    st.info("""
    **Next Steps**: 
    - Navigate to **Loyalty Pattern Analysis** to explore churn patterns
    - Use **Loyalty & LTV Predictor** to make real-time predictions
    - Check **Model Performance** to understand model accuracy
    """)

