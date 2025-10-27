import streamlit as st
import joblib
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("plotly").setLevel(logging.ERROR)

# Configurazione pagina
st.set_page_config(
    page_title="Customer Loyalty & LTV Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento modelli


@st.cache_resource
def load_all_models():
    """Load all ML models with error handling"""
    try:
        if os.path.exists('models/churn_model.pkl'):
            churn_model = joblib.load('models/churn_model.pkl')
        else:
            churn_model = None

        if os.path.exists('models/ltv_model.pkl'):
            ltv_model = joblib.load('models/ltv_model.pkl')
        else:
            ltv_model = None

        if os.path.exists('models/cluster_model.pkl'):
            cluster_model = joblib.load('models/cluster_model.pkl')
        else:
            cluster_model = None

        return churn_model, ltv_model, cluster_model
    except Exception:
        st.sidebar.warning(
            "Some models not found. Please run the notebooks to train them."
        )
        return None, None, None

# Caricamento dati


@st.cache_data
def load_customer_data():
    """Load customer dataset with error handling"""
    try:
        from src.data_loader import load_data
        df = load_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Sidebar navigazione
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Project Overview",
        "Loyalty Pattern Analysis",
        "Loyalty & LTV Predictor",
        "Hypothesis Validation",
        "Model Performance",
        "Retention ROI Analysis"
    ],
    index=0
)

# Caricamento iniziale
df = load_customer_data()

if df is not None:
    churn_model, ltv_model, cluster_model = load_all_models()

    # Routing delle pagine
    if page == "Project Overview":
        import app_pages.page_1 as page1
        page1.render(df)

    elif page == "Loyalty Pattern Analysis":
        import app_pages.page_2 as page2
        page2.render(df)

    elif page == "Loyalty & LTV Predictor":
        import app_pages.page_3 as page3
        page3.render(df, churn_model, ltv_model, cluster_model)

    elif page == "Hypothesis Validation":
        import app_pages.page_4 as page4
        page4.render(df)

    elif page == "Model Performance":
        import app_pages.page_5 as page5
        page5.render()

    elif page == "Retention ROI Analysis":
        import app_pages.page_6 as page6
        page6.render(df, ltv_model)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Customer Loyalty & LTV Predictor")
    st.sidebar.markdown("Built for telecom retention")
    st.sidebar.markdown("**Portfolio Project 5**: Predictive Analytics")

    # Dataset info
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.markdown(f"**Records**: {len(df):,}")
        st.sidebar.markdown(f"**Features**: {len(df.columns)}")
        if 'Churn' in df.columns:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.sidebar.markdown(f"**Churn Rate**: {churn_rate:.1f}%")
else:
    st.error(
        "Error loading dataset. Please ensure "
        "`WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the `data/` folder."
    )
    st.info(
        "Download the dataset from "
        "[Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)"
    )
