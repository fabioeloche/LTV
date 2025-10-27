"""
Visualizations Module
Creating interactive and static visualizations for the dashboard
LO6: 6.4, 6.5 - Interactive Plotly Visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import os

# Suppress Plotly deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

# Set Plotly configuration to suppress warnings
os.environ['PLOTLY_RENDERER'] = 'notebook'


def create_correlation_heatmap(df, method='spearman'):
    """
    Create interactive correlation heatmap
    LO6: 6.5 - Interactive Plotly Plot

    Args:
        df (pd.DataFrame): Dataset
        method (str): Correlation method ('spearman' or 'pearson')

    Returns:
        plotly.graph_objects.Figure: Interactive heatmap
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate correlation
    corr_matrix = df[numerical_cols].corr(method=method)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f"{method.capitalize()} Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    # Set size using config instead of deprecated width/height
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_churn_by_contract_plot(df):
    """
    Create barplot showing churn rate by contract type
    LO6: 6.5 - Interactive Plotly Plot

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        plotly.graph_objects.Figure: Interactive barplot
    """
    # Calculate churn rate by contract
    churn_by_contract = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).reset_index()
    churn_by_contract.columns = ['Contract', 'ChurnRate']

    fig = px.bar(
        churn_by_contract,
        x='Contract',
        y='ChurnRate',
        title='Churn Rate by Contract Type',
        labels={'ChurnRate': 'Churn Rate (%)', 'Contract': 'Contract Type'},
        color='ChurnRate',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        xaxis_title="Contract Type",
        yaxis_title="Churn Rate (%)",
        showlegend=False
    )

    return fig


def create_ltv_distribution_plot(df):
    """
    Create LTV distribution histogram
    LO6: 6.5 - Interactive Plotly Plot

    Args:
        df (pd.DataFrame): Customer dataset

    Returns:
        plotly.graph_objects.Figure: Interactive histogram
    """
    df_no_churn = df[df['Churn'] == 'No'].copy()
    df_no_churn['LTV'] = df_no_churn['MonthlyCharges'] * df_no_churn['tenure']

    fig = px.histogram(
        df_no_churn,
        x='LTV',
        nbins=50,
        title='Customer Lifetime Value Distribution (Non-Churned Customers)',
        labels={'LTV': 'Lifetime Value ($)', 'count': 'Number of Customers'},
        color_discrete_sequence=['#1f77b4']
    )

    fig.add_vline(
        x=df_no_churn['LTV'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${df_no_churn['LTV'].mean():.2f}"
    )

    fig.update_layout(
        xaxis_title="Lifetime Value ($)",
        yaxis_title="Number of Customers"
    )

    return fig


def create_feature_importance_plot(model, feature_names, top_n=10):
    """
    Create feature importance barplot
    LO6: 6.5 - Interactive Plotly Plot

    Args:
        model: Trained tree-based model
        feature_names (list): List of feature names
        top_n (int): Number of top features to display

    Returns:
        plotly.graph_objects.Figure: Interactive barplot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(color=importances[indices], colorscale='Viridis')
    ))

    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_cluster_profile_plot(df, cluster_labels):
    """
    Create cluster profile visualization

    Args:
        df (pd.DataFrame): Customer dataset
        cluster_labels (array): Cluster assignments

    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
    """
    df_plot = df.copy()
    df_plot['Cluster'] = cluster_labels

    fig = px.scatter(
        df_plot,
        x='tenure',
        y='MonthlyCharges',
        color='Cluster',
        title='Customer Segmentation: Tenure vs Monthly Charges',
        labels={
            'tenure': 'Tenure (months)',
            'MonthlyCharges': 'Monthly Charges ($)'},
        hover_data=[
            'Contract',
            'Churn'])

    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_confusion_matrix_plot(cm, labels=['No Churn', 'Churn']):
    """
    Create confusion matrix heatmap

    Args:
        cm (array): Confusion matrix
        labels (list): Class labels

    Returns:
        plotly.graph_objects.Figure: Confusion matrix plot
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_roi_by_cluster_plot(df, cluster_labels):
    """
    Create ROI visualization by cluster

    Args:
        df (pd.DataFrame): Customer dataset
        cluster_labels (array): Cluster assignments

    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    df_plot = df.copy()
    df_plot['Cluster'] = cluster_labels
    df_plot['ChurnBinary'] = (df_plot['Churn'] == 'Yes').astype(int)

    # Calculate average churn rate and LTV by cluster
    cluster_stats = df_plot.groupby('Cluster').agg({
        'ChurnBinary': 'mean',
        'LTV': 'mean'
    }).reset_index()

    cluster_stats.columns = ['Cluster', 'ChurnRate', 'AvgLTV']
    cluster_stats['ROI'] = (cluster_stats['ChurnRate']
                            * cluster_stats['AvgLTV'] * 0.3) - 50

    fig = px.bar(
        cluster_stats,
        x='Cluster',
        y='ROI',
        title='Estimated ROI by Customer Cluster',
        labels={'ROI': 'Estimated ROI ($)', 'Cluster': 'Cluster ID'},
        color='ROI',
        color_continuous_scale='RdYlGn'
    )

    fig.update_layout(
        xaxis_title="Cluster ID",
        yaxis_title="Estimated ROI ($)"
    )

    return fig


def create_visualizations(df, models=None):
    """
    Create all visualizations for the dashboard

    Args:
        df (pd.DataFrame): Customer dataset
        models (dict): Trained models (optional)

    Returns:
        dict: Dictionary of all figures
    """
    visualizations = {}

    visualizations['correlation_heatmap'] = create_correlation_heatmap(df)
    visualizations['churn_by_contract'] = create_churn_by_contract_plot(df)
    visualizations['ltv_distribution'] = create_ltv_distribution_plot(df)

    if models and 'churn_model' in models:
        feature_names = df.drop(
            ['customerID', 'Churn', 'TotalCharges'], axis=1,
            errors='ignore').columns
        visualizations['feature_importance'] = create_feature_importance_plot(
            models['churn_model'], feature_names
        )

    return visualizations
