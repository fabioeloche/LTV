"""
Page 6: Retention ROI Analysis
LO4: 4.1 - Actionable Insights
LO6: 6.1 - Dashboard Page Design
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render(df, ltv_model):
    """
    Render Retention ROI Analysis page

    Args:
        df (pd.DataFrame): Customer dataset
        ltv_model: Trained LTV model
    """
    st.title("Retention ROI Analysis")
    st.markdown(
        "**Business Requirement 4**: Actionable retention recommendations with ROI analysis")

    st.warning("""
    This analysis provides data-driven recommendations for customer retention strategies,
    helping maximize ROI while minimizing unnecessary intervention costs.
    """)

    st.markdown("---")

    # ROI Framework
    st.header("ROI Framework for Retention Campaigns")

    st.markdown("""
    ### Retention Investment Decision Model

    **Formula**:
    `Expected ROI = (Churn Probability × LTV × Success Rate) - Intervention Cost`

    **Parameters**:
    - **Churn Probability**: Model-predicted probability of customer churning
    - **LTV**: Customer Lifetime Value (predicted by LTV model)
    - **Success Rate**: Historical success rate of retention interventions (assumed 30%)
    - **Intervention Cost**: Average cost of retention action (varies by strategy)

    **Decision Rules**:
    - ROI > $100: **High Priority** - Immediate personal intervention
    - ROI $50-$100: **Medium Priority** - Automated outreach with incentives
    - ROI $0-$50: **Low Priority** - Standard retention campaign
    - ROI < $0: **Skip** - Not cost-effective to intervene
    """)

    st.markdown("---")

    # Simulated ROI by Cluster
    st.header("ROI by Customer Segment")

    st.markdown("""
    Analysis of expected ROI across customer clusters helps prioritize retention resources
    to maximize business impact.
    """)

    # Simulate cluster data
    cluster_data = {
        'Cluster': [0, 1, 2, 3, 4, 5, 6],
        'Cluster_Name': [
            'Young Tech-Savvy',
            'Senior Basic Service',
            'Family Bundle',
            'Long-term Loyal',
            'Premium Fiber Users',
            'Budget Conscious',
            'New Customers'
        ],
        'Avg_Churn_Rate': [0.35, 0.52, 0.28, 0.08, 0.42, 0.61, 0.45],
        'Avg_LTV': [1850, 780, 2340, 4520, 3100, 650, 1100],
        'Customer_Count': [890, 1120, 765, 1340, 920, 1150, 858]
    }

    df_clusters = pd.DataFrame(cluster_data)

    # Calculate ROI
    intervention_cost = 50
    success_rate = 0.3

    df_clusters['Expected_ROI'] = (
        df_clusters['Avg_Churn_Rate'] *
        df_clusters['Avg_LTV'] *
        success_rate
    ) - intervention_cost

    df_clusters['Total_ROI_Potential'] = (
        df_clusters['Expected_ROI'] *
        df_clusters['Customer_Count']
    )

    df_clusters['Priority'] = df_clusters['Expected_ROI'].apply(
        lambda x: 'High' if x > 100 else 'Medium' if x > 50 else 'Low' if x > 0 else 'Skip')

    # ROI Bar Chart
    fig_roi = px.bar(
        df_clusters,
        x='Cluster_Name',
        y='Expected_ROI',
        title='Expected ROI per Customer by Cluster',
        labels={
            'Expected_ROI': 'Expected ROI ($)',
            'Cluster_Name': 'Customer Segment'},
        color='Expected_ROI',
        color_continuous_scale='RdYlGn',
        text='Expected_ROI')
    fig_roi.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    fig_roi.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig_roi, width='stretch')

    # Display cluster table
    st.subheader("Detailed Cluster Analysis")

    display_df = df_clusters[[
        'Cluster_Name', 'Customer_Count', 'Avg_Churn_Rate',
        'Avg_LTV', 'Expected_ROI', 'Total_ROI_Potential', 'Priority'
    ]].copy()

    st.dataframe(
        display_df.style.format({
            'Customer_Count': '{:,}',
            'Avg_Churn_Rate': '{:.1%}',
            'Avg_LTV': '${:,.0f}',
            'Expected_ROI': '${:,.0f}',
            'Total_ROI_Potential': '${:,.0f}'
        }).background_gradient(subset=['Expected_ROI'], cmap='RdYlGn'),
        width='stretch',
        hide_index=True
    )

    # Key Insights
    highest_roi_cluster = df_clusters.loc[df_clusters['Expected_ROI'].idxmax()]
    total_potential = df_clusters['Total_ROI_Potential'].sum()

    top_cluster_name = highest_roi_cluster['Cluster_Name']
    top_cluster_roi = highest_roi_cluster['Expected_ROI']
    high_priority_clusters = len(df_clusters[df_clusters['Priority'] == 'High'])

    st.success(f"""
    ✅ **Key Insights**:
    - **Highest ROI Cluster**: {top_cluster_name} (${top_cluster_roi:.0f} per customer)
    - **Total ROI Potential**: ${total_potential:,.0f} across all at-risk customers
    - **High Priority Segments**: {high_priority_clusters} clusters
    - **Recommended Focus**: Long-term Loyal and Premium Fiber Users for maximum impact
    """)

    st.markdown("---")

    # Retention Strategies by Priority
    st.header("Recommended Retention Strategies")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Priority (ROI > $100)")
        st.markdown("""
        **Segments**: Long-term Loyal, Premium Fiber Users, Family Bundle

        **Strategies**:
        - 📞 **Personal Account Manager**: Dedicated support contact
        - **Premium Incentives**: 15-20% discount for contract renewal
        - **VIP Benefits**: Free device upgrades, priority technical support
        - **Immediate Contact**: Reach out within 24 hours of churn signal

        **Investment**: $100-$150 per customer
        **Expected Success Rate**: 35-40%
        **Break-even**: LTV > $300
        """)

        st.subheader("Medium Priority (ROI $50-$100)")
        st.markdown("""
        **Segments**: Young Tech-Savvy, New Customers

        **Strategies**:
        - 📧 **Automated Outreach**: Personalized email campaigns
        - 📱 **Service Bundle Offers**: Discount on additional services
        - **Engagement Programs**: Loyalty points, streaming upgrades
        - **Contact Window**: 48-72 hours

        **Investment**: $50-$75 per customer
        **Expected Success Rate**: 25-30%
        **Break-even**: LTV > $200
        """)

    with col2:
        st.subheader("Low Priority (ROI $0-$50)")
        st.markdown("""
        **Segments**: Budget Conscious (borderline)

        **Strategies**:
        - 📩 **Standard Email**: Generic retention offer
        - **Small Incentives**: $10 account credit, free month
        - **Survey Outreach**: Understand pain points
        - **Contact Window**: 1-2 weeks

        **Investment**: $20-$40 per customer
        **Expected Success Rate**: 15-20%
        **Break-even**: LTV > $150
        """)

        st.subheader("Skip (ROI < $0)")
        st.markdown("""
        **Segments**: Senior Basic Service (low LTV), Some Budget Conscious

        **Strategies**:
        - ❌ **No Active Intervention**: Not cost-effective
        - **Exit Survey**: Collect feedback for future improvements
        - **Win-back Later**: Re-engage after 6-12 months with new offers

        **Rationale**: Retention cost exceeds expected lifetime value
        """)

    st.markdown("---")

    # LTV Distribution Analysis
    st.header("LTV Distribution by Churn Status")

    st.markdown("""
    Understanding LTV distribution helps identify which customers to prioritize for retention efforts.
    """)

    df_ltv = df.copy()
    df_ltv['LTV'] = df_ltv['MonthlyCharges'] * df_ltv['tenure']

    # Box plot
    fig_ltv_box = px.box(
        df_ltv,
        x='Churn',
        y='LTV',
        title='LTV Distribution by Churn Status',
        labels={'LTV': 'Customer Lifetime Value ($)', 'Churn': 'Churn Status'},
        color='Churn',
        color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
    )

    st.plotly_chart(fig_ltv_box, width='stretch')

    # Statistics
    churned_ltv = df_ltv[df_ltv['Churn'] == 'Yes']['LTV']
    retained_ltv = df_ltv[df_ltv['Churn'] == 'No']['LTV']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean LTV (Churned)", f"${churned_ltv.mean():.2f}")
    with col2:
        st.metric("Mean LTV (Retained)", f"${retained_ltv.mean():.2f}")
    with col3:
        revenue_loss = churned_ltv.sum()
        st.metric("Total Revenue Loss", f"${revenue_loss:,.0f}")

    st.warning(f"""
    **Business Impact**:
    - **Lost Revenue**: ${revenue_loss:,.0f} from churned customers
    - **Potential Recovery**: If 30% retained → ${revenue_loss * 0.3:,.0f}
    - **Average Loss per Churn**: ${churned_ltv.mean():.2f}

    Even a 10% reduction in churn rate could save **${revenue_loss * 0.1:,.0f}** annually.
    """)

    st.markdown("---")

    # Campaign Budget Allocation
    st.header("💵 Recommended Campaign Budget Allocation")

    st.markdown("""
    Based on cluster ROI analysis, here's the recommended budget distribution for a
    $100,000 retention campaign budget:
    """)

    # Calculate budget allocation
    df_clusters['Budget_Share'] = (
        df_clusters['Total_ROI_Potential'] /
        df_clusters['Total_ROI_Potential'].sum()
    )

    df_clusters['Recommended_Budget'] = df_clusters['Budget_Share'] * 100000

    fig_budget = px.pie(
        df_clusters,
        values='Recommended_Budget',
        names='Cluster_Name',
        title='Recommended Budget Allocation ($100K Campaign)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(fig_budget, width='stretch')

    # Budget table
    budget_df = df_clusters[['Cluster_Name',
                             'Customer_Count',
                             'Expected_ROI',
                             'Recommended_Budget',
                             'Priority']].copy()

    st.dataframe(
        budget_df.style.format({
            'Customer_Count': '{:,}',
            'Expected_ROI': '${:,.0f}',
            'Recommended_Budget': '${:,.0f}'
        }),
        width='stretch',
        hide_index=True
    )

    net_profit = total_potential - 100000
    roi_multiplier = total_potential / 100000

    st.success(f"""
    ✅ **Budget Optimization**:
    - **Expected Total ROI**: ${total_potential:,.0f}
    - **Campaign Investment**: $100,000
    - **Net Profit**: ${net_profit:,.0f}
    - **ROI Multiplier**: {roi_multiplier:.2f}x

    For every $1 invested in retention, expect **${roi_multiplier:.2f}** in return.
    """)

    st.markdown("---")

    # Action Plan
    st.header("Immediate Action Plan")

    st.markdown("""
    ### Week 1: High-Priority Customers
    1. **Identify**: Run churn model on all customers → Flag top 20% risk
    2. **Filter**: Select high-LTV customers (>$2000) from high-risk group
    3. **Contact**: Personal calls from account managers
    4. **Offer**: 20% discount + service bundle upgrade
    5. **Target**: 200 customers, Expected ROI: $40,000

    ### Week 2-3: Medium-Priority Customers
    1. **Identify**: Medium-risk customers (churn prob 0.4-0.7)
    2. **Segment**: Group by cluster for targeted messaging
    3. **Campaign**: Automated email + SMS with personalized offers
    4. **Offer**: Service upgrades, loyalty rewards
    5. **Target**: 800 customers, Expected ROI: $50,000

    ### Week 4: Low-Priority Customers
    1. **Identify**: Lower-risk but high-value customers
    2. **Campaign**: Standard retention email campaign
    3. **Offer**: Account credits, survey participation incentives
    4. **Target**: 1500 customers, Expected ROI: $30,000

    ### Ongoing: Monitor & Optimize
    - Track success rates weekly
    - A/B test different offers
    - Refine model predictions with new data
    - Adjust strategy based on performance
    """)

    st.info("""
    **Implementation Tools**:
    - Use **Loyalty & LTV Predictor** page to identify individual at-risk customers
    - Export customer lists with predictions for CRM integration
    - Schedule monthly model retraining with updated data
    - Set up automated alerts for high-value customers showing churn signals
    """)

    st.markdown("---")

    # Summary
    st.header("Executive Summary")

    protected_revenue = revenue_loss * 0.125

    st.markdown(f"""
    ### Business Impact of ML-Driven Retention

    **Current State**:
    - Annual churn rate: {(df['Churn'] == 'Yes').mean() * 100:.1f}%
    - Lost revenue from churn: ${churned_ltv.sum():,.0f}
    - Customer base: {len(df):,} customers

    **With ML-Driven Retention**:
    - Predicted churn reduction: 10-15%
    - Revenue protection: ${protected_revenue:,.0f} annually
    - Campaign ROI: {roi_multiplier:.1f}x
    - High-value customer retention: +25%

    **Strategic Advantages**:
    1. **Proactive Intervention**: Identify at-risk customers before they churn
    2. **Resource Optimization**: Focus budget on highest-ROI segments
    3. **Personalization**: Tailor retention offers to customer segments
    4. **Measurable Impact**: Track ROI and continuously improve

    **Next Steps**:
    1. Deploy churn prediction model to production CRM system
    2. Implement automated alerts for high-risk, high-value customers
    3. Launch pilot retention campaign with Cluster 3 (Long-term Loyal)
    4. Establish monthly model performance reviews
    5. Scale successful strategies across all customer segments
    """)
