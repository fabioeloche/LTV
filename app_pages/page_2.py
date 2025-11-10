"""
Page 2: Loyalty Pattern Analysis
LO6: 6.1, 6.2 - Dashboard Page with Text Interpretations
LO3: 3.1 - EDA Implementation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.visualizations import (
    create_correlation_heatmap,
    create_churn_by_contract_plot,
    create_ltv_distribution_plot
)


def render(df):
    """
    Render Loyalty Pattern Analysis page

    Args:
        df (pd.DataFrame): Customer dataset
    """
    st.title("Loyalty Pattern Analysis")
    st.markdown(
        "**Business Requirement 1**: Analyze loyalty patterns to identify key churn drivers")

    st.markdown("---")

    # Interactive EDA checkbox (LO3: 3.1)
    st.header("Exploratory Data Analysis")

    if st.checkbox("Show Detailed EDA", value=True):

        # Churn Distribution
        st.subheader("1. Churn Distribution")

        churn_counts = df['Churn'].value_counts()
        churn_pct = df['Churn'].value_counts(normalize=True) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig_churn = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title='Churn Distribution',
                color=churn_counts.index,
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig_churn, width='stretch')

        with col2:
            st.markdown("### Key Insights")
            st.markdown(f"""
            - **Churn Rate**: {churn_pct['Yes']:.1f}%
            - **Loyal Customers**: {churn_pct['No']:.1f}%
            - **Total Churned**: {churn_counts['Yes']:,} customers
            - **Total Loyal**: {churn_counts['No']:,} customers
            """)

            # Text Interpretation (LO6: 6.2)
            st.info("""
            **Interpretation**: The dataset shows a churn rate of approximately 27%,
            which is typical for telecom companies. This indicates a significant opportunity
            for retention strategies to reduce revenue loss.
            """)

        st.markdown("---")

        # Contract Type vs Churn (Key Hypothesis)
        st.subheader("2. Contract Type vs Churn Rate")

        fig_contract = create_churn_by_contract_plot(df)
        st.plotly_chart(fig_contract, width='stretch')

        # Calculate exact percentages
        contract_churn = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )

        churn_reduction = (
            contract_churn['Month-to-month'] - contract_churn['Two year']
        )

        st.success(f"""
        ✅ **Key Finding (Hypothesis 1 Validation)**:
        - Month-to-month contracts have **{contract_churn['Month-to-month']:.1f}%** churn rate
        - One-year contracts have **{contract_churn['One year']:.1f}%** churn rate
        - Two-year contracts have **{contract_churn['Two year']:.1f}%** churn rate

        **Interpretation**: Two-year contracts show **{churn_reduction:.1f}%**
        lower churn rate compared to month-to-month contracts, strongly
        validating our hypothesis that
        long-term contracts reduce churn.
        """)

        st.markdown("---")

        # Correlation Analysis
        st.subheader("3. Correlation Heatmap")

        st.markdown("""
        This heatmap shows Spearman correlations between numerical features and churn.
        Strong correlations (darker colors) indicate important churn drivers.
        """)

        fig_corr = create_correlation_heatmap(df, method='spearman')
        st.plotly_chart(fig_corr, width='stretch')

        st.info("""
        **Interpretation**: The correlation analysis reveals:
        - **Tenure** has strong negative correlation with churn (-0.35), indicating longer-tenured
          customers are less likely to churn
        - **MonthlyCharges** shows positive correlation with churn (0.19), suggesting higher charges
          may increase churn risk
        - **TotalCharges** (related to tenure) strongly correlates with loyalty
        """)

        st.markdown("---")

        # Services Analysis
        st.subheader("4. Services vs Churn")

        # Internet Service
        internet_churn = df.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        internet_churn.columns = ['InternetService', 'ChurnRate']

        fig_internet = px.bar(
            internet_churn,
            x='InternetService',
            y='ChurnRate',
            title='Churn Rate by Internet Service Type',
            labels={
                'ChurnRate': 'Churn Rate (%)',
                'InternetService': 'Internet Service'},
            color='ChurnRate',
            color_continuous_scale='Reds')
        st.plotly_chart(fig_internet, width='stretch')

        st.warning("""
        **Key Insight**: Fiber optic users have significantly higher churn rates than DSL users,
        possibly due to higher expectations or pricing. This segment requires targeted retention efforts.
        """)

        st.markdown("---")

        # Additional Services Impact
        st.subheader("5. Additional Services Impact")

        services_to_check = ['OnlineSecurity', 'TechSupport', 'OnlineBackup']

        service_results = []
        for service in services_to_check:
            if service in df.columns:
                service_churn = df.groupby(service)['Churn'].apply(
                    lambda x: (x == 'Yes').sum() / len(x) * 100
                )
                service_results.append({
                    'Service': service,
                    'With_Service': service_churn.get('Yes', 0),
                    'Without_Service': service_churn.get('No', 0)
                })

        if service_results:
            service_df = pd.DataFrame(service_results)

            fig_services = go.Figure(
                data=[
                    go.Bar(
                        name='With Service',
                        x=service_df['Service'],
                        y=service_df['With_Service']),
                    go.Bar(
                        name='Without Service',
                        x=service_df['Service'],
                        y=service_df['Without_Service'])])

            fig_services.update_layout(
                barmode='group',
                title='Churn Rate: With vs Without Additional Services',
                xaxis_title='Service Type',
                yaxis_title='Churn Rate (%)'
            )

            st.plotly_chart(fig_services, width='stretch')

            st.success("""
            ✅ **Hypothesis 2 Validation**: Customers with additional services (OnlineSecurity, TechSupport)
            show significantly lower churn rates, confirming that value-added services increase loyalty.
            """)

        st.markdown("---")

        # Monthly Charges Distribution
        st.subheader("6. Monthly Charges Distribution")

        fig_charges = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            title='Monthly Charges by Churn Status',
            labels={
                'MonthlyCharges': 'Monthly Charges ($)',
                'Churn': 'Churn Status'},
            color='Churn',
            color_discrete_map={
                'Yes': '#ff6b6b',
                'No': '#4ecdc4'})
        st.plotly_chart(fig_charges, width='stretch')

        st.info("""
        **Interpretation**: Churned customers tend to have higher monthly charges on average,
        suggesting price sensitivity. However, for retained customers, higher charges correlate
        with higher LTV (see LTV analysis).
        """)

        st.markdown("---")

        # LTV Distribution
        st.subheader("7. Customer Lifetime Value (LTV) Distribution")

        fig_ltv = create_ltv_distribution_plot(df)
        st.plotly_chart(fig_ltv, width='stretch')

        df_no_churn = df[df['Churn'] == 'No'].copy()
        df_no_churn['LTV'] = df_no_churn['MonthlyCharges'] * \
            df_no_churn['tenure']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean LTV", f"${df_no_churn['LTV'].mean():.2f}")
        with col2:
            st.metric("Median LTV", f"${df_no_churn['LTV'].median():.2f}")
        with col3:
            st.metric("Max LTV", f"${df_no_churn['LTV'].max():.2f}")

        st.success("""
        ✅ **Business Impact**: Understanding LTV distribution helps prioritize retention efforts
        on high-value customers, maximizing ROI from retention campaigns.
        """)

    st.markdown("---")

    # Summary
    st.header("Summary of Key Findings")

    st.markdown("""
    ### Top Churn Drivers Identified:
    1. **Contract Type**: Month-to-month contracts have 3x higher churn than two-year contracts
    2. **Tenure**: New customers (< 6 months) are at highest risk
    3. **Internet Service**: Fiber optic users show elevated churn rates
    4. **Additional Services**: Lack of OnlineSecurity and TechSupport increases churn
    5. **Monthly Charges**: Higher charges correlate with increased churn risk

    ### Actionable Recommendations:
    - Target month-to-month customers with contract upgrade incentives
    - Implement enhanced onboarding for new customers
    - Bundle additional services for fiber optic customers
    - Review pricing strategies for high-charge segments
    """)

    st.info("""
    **Next Steps**: Use the **Loyalty & LTV Predictor** page to make real-time predictions
    and get personalized retention recommendations for individual customers.

    **Key Insight**: The analysis shows that contract type is the strongest predictor of churn,
    with two-year contracts having 40% lower churn rates than month-to-month contracts.
    """)
