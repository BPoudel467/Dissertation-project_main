"""
Streamlit UI for IDS Comparative Study
=======================================
A web-based interface for running the intrusion detection pipeline
and visualizing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="IDS Comparative Study",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def get_output_path(subpath: str) -> Path:
    """Get output file path."""
    return PROJECT_ROOT / "outputs" / subpath


def get_data_path(subpath: str) -> Path:
    """Get data file path."""
    return PROJECT_ROOT / "data" / subpath


# ==================== SIDEBAR ====================
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title(" IDS Study")
    st.sidebar.markdown("---")
    
    # Navigation menu
    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Pipeline", "Results", "Model Comparison", "Visualizations", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("📊 IDS Comparative Study\nMachine Learning Pipeline")
    
    return menu


# ==================== HOME PAGE ====================
def render_home():
    """Render the home page."""
    st.markdown('<p class="main-header"> Intrusion Detection System Comparative Study</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Supervised Models", "3", "RF, SVM, XGBoost")
    with col2:
        st.metric("Unsupervised Models", "2", "K-Means, Autoencoder")
    with col3:
        st.metric("Pipeline Steps", "6", "Full automation")
    
    st.markdown("---")
    
    # Project overview
    st.subheader("📋 Project Overview")
    st.markdown("""
    This application implements a comparative study of machine learning models for 
    **network intrusion detection** using the 5G-NIDD dataset.
    
    ### Features:
    - **Supervised Learning**: Random Forest, SVM, XGBoost
    - **Unsupervised Learning**: K-Means Clustering, Autoencoder
    - **Data Preprocessing**: Cleaning, SMOTE for imbalance handling
    - **Feature Selection**: Random Forest based selection
    - **Comprehensive Evaluation**: Metrics, plots, and comparisons
    """)
    
    # Quick actions
    st.subheader(" Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶ Run Full Pipeline", use_container_width=True):
            st.session_state['run_pipeline'] = True
            st.rerun()
    
    with col2:
        if st.button(" View Results", use_container_width=True):
            st.session_state['page'] = 'Results'
            st.rerun()


# ==================== RUN PIPELINE PAGE ====================
def render_run_pipeline():
    """Render the pipeline execution page."""
    st.markdown('<p class="sub-header"> Run Pipeline</p>', unsafe_allow_html=True)
    
    st.info("💡 Click the button below to run the complete ML pipeline. This may take several minutes.")
    
    # Pipeline steps info
    st.subheader("Pipeline Steps:")
    
    steps = [
        ("1", "Load and Combine UNSW-NB15", "Load raw data files and combine them"),
        ("2", "Clean Dataset", "Remove duplicates, handle missing values"),
        ("3", "Preprocess and Split", "Encode categorical, scale features, train/test split"),
        ("4", "Handle Class Imbalance", "Apply SMOTE to balance classes"),
        ("5", "Feature Selection", "Select important features using Random Forest"),
        ("6", "Train Supervised Models", "Train and evaluate RF, SVM, XGBoost"),
    ]
    
    for num, title, desc in steps:
        with st.expander(f"Step {num}: {title}"):
            st.write(desc)
    
    st.markdown("---")
    
    # Run button
    if st.button(" Execute Full Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running pipeline... This may take several minutes."):
            try:
                # Import and run main
                from src.utils.common import ensure_directories, set_seed
                from src.data.load_unsw import load_and_combine_unsw
                from src.data.clean_data import clean_unsw_data
                from src.data.preprocess import preprocess_and_save
                from src.data.imbalance import apply_smote
                from src.features.feature_selection import select_features_with_rf
                from src.models.train_supervised import run_supervised_training
                
                # Run pipeline
                ensure_directories()
                set_seed()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1
                status_text.text("Step 1: Loading and combining UNSW-NB15 files...")
                df = load_and_combine_unsw()
                progress_bar.progress(16)
                
                # Step 2
                status_text.text("Step 2: Cleaning dataset...")
                df_clean = clean_unsw_data(df)
                progress_bar.progress(32)
                
                # Step 3
                status_text.text("Step 3: Preprocessing and splitting...")
                preprocess_and_save(df_clean)
                progress_bar.progress(48)
                
                # Step 4
                status_text.text("Step 4: Handling class imbalance...")
                apply_smote()
                progress_bar.progress(64)
                
                # Step 5
                status_text.text("Step 5: Feature selection...")
                select_features_with_rf()
                progress_bar.progress(80)
                
                # Step 6
                status_text.text("Step 6: Training supervised models...")
                run_supervised_training()
                progress_bar.progress(100)
                
                status_text.success(" Pipeline completed successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f" Error running pipeline: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# ==================== RESULTS PAGE ====================
def render_results():
    """Render the results page."""
    st.markdown('<p class="sub-header"> Results</p>', unsafe_allow_html=True)
    
    tables_path = get_output_path("tables")
    
    # Check for results files
    supervised_results = tables_path / "supervised_results.csv"
    unsupervised_results = tables_path / "unsupervised_results.csv"
    
    tab1, tab2 = st.tabs(["Supervised Models", "Unsupervised Models"])
    
    with tab1:
        if supervised_results.exists():
            df = pd.read_csv(supervised_results)
            st.dataframe(df, use_container_width=True)
            
            # Display metrics
            st.subheader("Model Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            for idx, row in df.iterrows():
                with st.expander(f"{row.get('Model', 'Unknown')}"):
                    st.write(f"**Accuracy:** {row.get('Accuracy', 'N/A')}")
                    st.write(f"**Precision:** {row.get('Precision', 'N/A')}")
                    st.write(f"**Recall:** {row.get('Recall', 'N/A')}")
                    st.write(f"**F1-Score:** {row.get('F1-Score', 'N/A')}")
        else:
            st.warning(" No supervised results found. Please run the pipeline first.")
            st.info(" Go to 'Run Pipeline' to execute the ML pipeline.")
    
    with tab2:
        if unsupervised_results.exists():
            df = pd.read_csv(unsupervised_results)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(" No unsupervised results found. Please run the pipeline first.")

# ==================== MODEL COMPARISON PAGE ====================
def render_model_comparison():
    """Render the model comparison page."""
    st.markdown('<p class="sub-header"> Model Comparison</p>', unsafe_allow_html=True)
    
    tables_path = get_output_path("tables")
    comparison_file = tables_path / "model_comparison.csv"
    
    if comparison_file.exists():
        df = pd.read_csv(comparison_file)
        
        # Display comparison table
        st.subheader("Model Performance Comparison")
        st.dataframe(df, use_container_width=True)
        
        # Create comparison charts
        st.subheader("📈 Performance Visualization")
        
        # Try to load and display comparison figure
        figures_path = get_output_path("figures")
        comparison_fig = figures_path / "model_performance_comparison.png"
        
        if comparison_fig.exists():
            st.image(str(comparison_fig), caption="Model Performance Comparison")
        else:
            st.info("📊 Comparison figure not yet generated.")
    else:
        st.warning("⚠️ No comparison results found. Please run the pipeline first.")
        st.info("💡 Go to 'Run Pipeline' to execute the ML pipeline.")


# ==================== VISUALIZATIONS PAGE ====================
def render_visualizations():
    """Render the visualizations page."""
    st.markdown('<p class="sub-header"> Visualizations</p>', unsafe_allow_html=True)
    
    figures_path = get_output_path("figures")
    
    # Check what figures exist
    available_figures = []
    
    if figures_path.exists():
        for f in figures_path.glob("*.png"):
            available_figures.append(f.name)
    
    if available_figures:
        # Create tabs for each figure
        tabs = st.tabs(available_figures)
        
        for i, (tab, fig_name) in enumerate(zip(tabs, available_figures)):
            with tab:
                fig_path = figures_path / fig_name
                st.image(str(fig_path), caption=fig_name)
    else:
        st.warning("⚠️ No visualizations found. Please run the pipeline first.")
        st.info("💡 Go to 'Run Pipeline' to execute the ML pipeline.")


# ==================== ABOUT PAGE ====================
def render_about():
    """Render the about page."""
    st.markdown('<p class="sub-header">ℹAbout</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ##  Intrusion Detection System Comparative Study
    
    ### Project Description
    This project implements a comparative study of machine learning models for 
    network intrusion detection using the **5G-NIDD** dataset.
    
    ### Dataset
    - **5G-NIDD**: A modern network intrusion dataset
    - 49 features describing network flows
    - Multiple attack categories
    
    ### Machine Learning Models
    
    #### Supervised Models
    - **Random Forest**: Ensemble learning method
    - **Support Vector Machine (SVM)**: Classification algorithm
    - **XGBoost**: Gradient boosting algorithm
    
    #### Unsupervised Models
    - **K-Means Clustering**: Partitioning method
    - **Autoencoder**: Neural network for anomaly detection
    
    ### Pipeline Steps
    1. Data Loading & Combination
    2. Data Cleaning
    3. Preprocessing & Splitting
    4. Class Imbalance Handling (SMOTE)
    5. Feature Selection
    6. Model Training & Evaluation
    
    ### Technologies Used
    - Python 3.x
    - TensorFlow / Keras
    - Scikit-learn
    - XGBoost
    - imbalanced-learn
    - Streamlit (for UI)
    
    ### Author
    Master's Dissertation in Computer Science
    """)
    
    st.markdown("---")
    st.caption("© 2024 IDS Comparative Study")


# ==================== MAIN APP ====================
def main():
    """Main application entry point."""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    st.session_state['page'] = current_page
    
    # Render current page
    if current_page == "Home":
        render_home()
    elif current_page == "Run Pipeline":
        render_run_pipeline()
    elif current_page == "Results":
        render_results()
    elif current_page == "Model Comparison":
        render_model_comparison()
    elif current_page == "Visualizations":
        render_visualizations()
    elif current_page == "About":
        render_about()


if __name__ == "__main__":
    main()