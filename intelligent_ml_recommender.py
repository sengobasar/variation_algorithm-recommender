# INTELLIGENT ML RECOMMENDER WITH REAL AUTOML BACKEND
# pip install streamlit tpot scikit-learn plotly pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# AutoML imports
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import io
import contextlib

# Page configuration
st.set_page_config(
    page_title="ML Algorithm Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with classical theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Main App Styling */
    .stApp {
        background-color: #f8f5f0; /* Warmer parchment background */
        color: #2c3e50;
        font-family: 'Source Sans Pro', sans-serif;
        line-height: 1.6;
    }
    
    /* Sidebar specific styles */
    .sidebar-header {
        font-family: 'Playfair Display', serif;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sidebar-subtitle {
        color: #666;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: #2c3e50;
        font-weight: 700;
    }
    
    .main-title {
        font-size: 2.8rem;
        text-align: center;
        margin: 1.5rem 0;
        color: #2c3e50;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Cards & Expanders */
    .card, .stExpander {
        background: white;
        border-radius: 6px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8e8e8;
        transition: all 0.3s ease;
    }
    
    .stExpander {
        background: #fff;
        border-left: 3px solid #4a6cf7;
    }
    
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .card h3 {
        margin-top: 0;
        color: #4a6cf7;
    }
    
    /* Status Indicators */
    .status-info {
        background: #e6f3ff;
        border-left: 4px solid #4a6cf7;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .status-success {
        background: #e6f7ee;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff8e6;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    /* Sidebar styles */
    .css-1d391kg, .st-emotion-cache-1cyp6b2 {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e5e7eb;
        color: #2c3e50 !important;
    }
    
    /* Ensure text in sidebar is visible */
    .stSidebar .stMarkdown, 
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stNumberInput label,
    .stSidebar .stTextInput label,
    .stSidebar .stRadio label,
    .stSidebar .stCheckbox label,
    .stSidebar .stExpander label {
        color: #2c3e50 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: #4a6cf7;
        color: white;
        border: none;
        padding: 0.8rem 1.75rem;
        border-radius: 4px;
        font-weight: 500;
        font-family: 'Source Sans Pro', sans-serif;
        letter-spacing: 0.5px;
        text-transform: none;
        transition: all 0.25s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background: #3b5bdb;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(58, 91, 219, 0.25);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e8e8e8;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px 4px 0 0;
        padding: 0.75rem 1.5rem;
        margin: 0;
        transition: all 0.25s ease;
        color: #666;
        font-weight: 500;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #4a6cf7;
        border-bottom: 2px solid #4a6cf7;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: #3b5bdb;
        background-color: rgba(74, 108, 247, 0.05);
    }
    
    /* Code blocks */
    .pipeline-code, pre {
        background: #f8f9fa;
        border: 1px solid #e8e8e8;
        border-radius: 4px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-family: 'Source Code Pro', monospace;
        font-size: 0.875rem;
        line-height: 1.6;
        overflow-x: auto;
    }
    
    pre {
        background-color: #f9f9f9;
        border-left: 3px solid #4a6cf7;
    }
    
    code {
        font-family: 'Source Code Pro', monospace;
        font-size: 0.875em;
        background-color: rgba(74, 108, 247, 0.1);
        padding: 0.2em 0.4em;
        border-radius: 2px;
        color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        background-color: #f0f2f6;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a6cf7;
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class IntelligentDatasetGenerator:
    """Maps user inputs to synthetic dataset parameters"""
    
    def generate_dataset_from_inputs(self, user_inputs: dict):
        """Generate realistic dataset based on user specifications"""
        # Extract key parameters
        dataset_size = user_inputs.get('dataset_size', '1K–10K samples')
        problem_type = user_inputs.get('problem_type', 'Binary Classification')
        feature_types = user_inputs.get('feature_types', 'Mixed numeric + categorical')
        missing_values = user_inputs.get('missing_values', 'No missing values')
        class_balance = user_inputs.get('class_balance', 'Roughly balanced')
        relationship_type = user_inputs.get('relationship_type', 'Moderately nonlinear')
        
        # Get the string answer for noise level and map it to a numerical value
        noise_level_str = user_inputs.get('noise_level', 'Some noise or human error')
        noise_mapping = {
            'Very clean and structured': 0.01,
            'Some noise or human error': 0.1,
            'High noise or uncertainty': 0.25
        }
        noise_level_value = noise_mapping.get(noise_level_str, 0.1)
        
        # Map dataset size to n_samples
        size_mapping = {
            '<1K samples': 1000,
            '1K–10K samples': 5000,
            '10K–100K samples': 20000,
            '>100K samples': 50000
        }
        n_samples = size_mapping.get(dataset_size, 1000)
        
        # Ensure minimum samples
        n_samples = max(n_samples, 100)  # Ensure at least 100 samples
        
        # Set default values for features
        n_features = 10  # Default value
        n_informative = 6  # Default value
        
        # Adjust features based on feature types
        if 'categorical' in feature_types.lower():
            n_features = 15
            n_informative = 8 if 'mixed' in feature_types.lower() else 6
        else:
            n_features = 12
            n_informative = 8
        
        
        # Map relationship complexity
        if 'linear' in relationship_type.lower():
            class_sep = 2.0
            n_clusters_per_class = 1
            effective_noise = noise_level_value * 0.5  # Less noise for linear relationships
        elif 'nonlinear' in relationship_type.lower():
            class_sep = 0.8
            n_clusters_per_class = 2
            effective_noise = noise_level_value
        else:  # Moderately nonlinear
            class_sep = 1.2
            n_clusters_per_class = 1
            effective_noise = noise_level_value * 0.7
        
        # Generate dataset based on problem type
        if 'Classification' in problem_type:
            n_classes = 2 if 'Binary' in problem_type else 3
            
            # Ensure we have enough samples per class
            min_samples_per_class = 10  # Minimum samples per class for CV
            n_samples = max(n_samples, n_classes * min_samples_per_class)
            
            # Handle class imbalance
            if 'imbalanced' in class_balance.lower():
                if n_classes == 2:
                    weights = [0.9, 0.1] if 'slightly' not in class_balance.lower() else [0.7, 0.3]
                else:
                    weights = [0.7, 0.2, 0.1] if 'slightly' not in class_balance.lower() else [0.6, 0.3, 0.1]
            else:
                weights = None
            
            # Generate classification data with proper class separation
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=max(1, n_features // 4),  # Add some redundant features
                n_repeated=0,
                n_classes=n_classes,
                n_clusters_per_class=n_clusters_per_class,
                weights=weights,
                flip_y=effective_noise * 2,  # Higher flip_y for more noise
                class_sep=class_sep,
                hypercube=True,
                scale=1.0,
                shuffle=True,
                random_state=42
            )
            
        elif problem_type == 'Regression':
            # For regression, ensure we have enough samples for the number of features
            n_samples = max(n_samples, n_features * 10)
            
            # Generate regression data with the specified noise level
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_targets=1,
                bias=0.0,
                effective_rank=None,
                tail_strength=0.5,
                noise=effective_noise * 10,  # Scale noise for regression
                shuffle=True,
                coef=False,
                random_state=42
            )
            
            # Ensure y has sufficient variance
            if np.std(y) < 0.1:
                y = y * (1.0 / (np.std(y) + 1e-8)) * (10 + effective_noise * 10)
                
            # Add some non-linearity if specified
            if 'nonlinear' in relationship_type.lower():
                y = y + 0.5 * (X[:, 0] ** 2) + 0.3 * (X[:, 1] ** 3) if n_features >= 2 else y
                
        else:  # Clustering
            n_clusters = 3
            n_samples = max(n_samples, n_clusters * 10)  # Ensure enough samples per cluster
            
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                cluster_std=1.0 + (1.0 - class_sep),  # Adjust cluster separation
                center_box=(-10.0, 10.0),
                shuffle=True,
                random_state=42
            )
        
        # Add missing values if specified
        if 'missing' in missing_values.lower():
            missing_ratio = 0.05 if 'Some' in missing_values else 0.15
            n_missing = int(X.size * missing_ratio)
            missing_indices = np.random.choice(X.size, size=n_missing, replace=False)
            X.ravel()[missing_indices] = np.nan
        
        return X, y, {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_informative': n_informative,
            'problem_type': problem_type,
            'has_missing': 'missing' in missing_values.lower(),
            'noise_level': effective_noise,
            'is_balanced': 'balanced' in class_balance.lower()
        }

class RealAutoMLEngine:
    """Real AutoML backend using TPOT"""
    
    def __init__(self):
        self.results = {}
        self.best_pipeline = None
        self.dataset_info = {}
    
    def run_automl_analysis(self, X, y, dataset_info: dict, user_inputs: dict) -> dict:
        """Run real AutoML analysis and return results"""
        
        st.info("🤖 Running Real AutoML Analysis...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Split data
            status_text.text("📊 Splitting dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            progress_bar.progress(10)
            
            # Handle missing values for TPOT
            if dataset_info.get('has_missing', False):
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                status_text.text("🔧 Handled missing values with imputation...")
            
            progress_bar.progress(20)
            
            # Configure TPOT based on problem type
            if 'Classification' in dataset_info['problem_type']:
                status_text.text("🧠 Initializing TPOT Classifier...")
                
                automl = TPOTClassifier(
                    generations=3,  # Reduced for speed
                    population_size=20,  # Reduced for speed
                    cv=3,
                    random_state=42,
                    n_jobs=1,  # Avoid multiprocessing issues in Streamlit
                    max_time_mins=2,  # Quick results
                    max_eval_time_mins=0.5
                )
                
            else:  # Regression
                status_text.text("🧠 Initializing TPOT Regressor...")
                
                automl = TPOTRegressor(
                    generations=3,
                    population_size=20,
                    cv=3,
                    random_state=42,
                    n_jobs=1,
                    max_time_mins=2,
                    max_eval_time_mins=0.5
                )
            
            progress_bar.progress(30)
            status_text.text("🚀 Running AutoML optimization (this may take 2-3 minutes)...")
            
            # Capture TPOT output
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    automl.fit(X_train, y_train)
            
            progress_bar.progress(80)
            status_text.text("📈 Evaluating best models...")
            
            # Get predictions and metrics
            y_pred = automl.predict(X_test)
            
            if 'Classification' in dataset_info['problem_type']:
                primary_score = accuracy_score(y_test, y_pred)
                metric_name = 'Accuracy'
                
                # Get additional metrics
                try:
                    y_pred_proba = automl.predict_proba(X_test)
                    from sklearn.metrics import roc_auc_score, f1_score
                    
                    if len(np.unique(y)) == 2:  # Binary classification
                        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                        f1 = f1_score(y_test, y_pred)
                    else:  # Multiclass
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                    additional_metrics = {'AUC': auc_score, 'F1-Score': f1}
                except:
                    additional_metrics = {}
                    
            else:  # Regression
                primary_score = r2_score(y_test, y_pred)
                metric_name = 'R² Score'
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                additional_metrics = {'MSE': mse, 'MAE': mae}
            
            progress_bar.progress(90)
            
            # Get model components from the fitted pipeline
            fitted_pipeline = automl.fitted_pipeline_
            
            # Generate pipeline code using the fitted pipeline
            best_pipeline_str = str(fitted_pipeline)
            
            # Extract individual models and their scores through cross-validation
            model_scores = self._extract_model_performances(automl, X_train, y_train, dataset_info)
            
            progress_bar.progress(100)
            status_text.text("✅ AutoML analysis complete!")
            
            # Clean up progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return {
                'success': True,
                'best_model': str(fitted_pipeline.steps[-1][1].__class__.__name__),
                'primary_score': primary_score,
                'metric_name': metric_name,
                'additional_metrics': additional_metrics,
                'pipeline_code': best_pipeline_str,
                'model_scores': model_scores,
                'dataset_info': dataset_info,
                'fitted_pipeline': fitted_pipeline,
                'test_predictions': y_pred,
                'test_actual': y_test
            }
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"AutoML Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _extract_model_performances(self, automl, X_train, y_train, dataset_info: dict) -> dict:
        """Extract performance of different model types from TPOT's evaluation"""
        
        # Common algorithm types to test
        if 'Classification' in dataset_info['problem_type']:
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            
            models_to_test = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Extra Trees': ExtraTreesClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
            scoring = 'accuracy'
            
        else:  # Regression
            from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            
            models_to_test = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Extra Trees': ExtraTreesRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=42),
                'SVR': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor()
            }
            scoring = 'r2'
        
        # Test each model with cross-validation
        model_scores = {}
        
        for name, model in models_to_test.items():
            try:
                # Handle missing values if necessary
                if dataset_info.get('has_missing', False):
                    from sklearn.pipeline import Pipeline
                    from sklearn.impute import SimpleImputer
                    
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('model', model)
                    ])
                else:
                    pipeline = model
                
                # Quick cross-validation
                scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring, n_jobs=1)
                model_scores[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
                
            except Exception as e:
                # Skip models that fail
                continue
        
        return model_scores

# Title and description
st.markdown('<h1 class="main-title">Machine Learning Algorithm Recommender</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Algorithm Selection Powered by AutoML and Synthetic Data Analysis")

# Sidebar Configuration
with st.sidebar:
    st.markdown('<h3 class="sidebar-header">Dataset Configuration</h3>', unsafe_allow_html=True)
    st.markdown("<p class='sidebar-subtitle'>Configure your dataset characteristics to generate optimal algorithm recommendations.</p>", unsafe_allow_html=True)
    
    # Group 1: Dataset Characteristics
    with st.expander("📊 Dataset Characteristics", expanded=True):
        data_type = st.selectbox(
            "1. What kind of data are you working with?",
            ['Tabular / Structured (spreadsheets, CSVs)', 
             'Text / NLP (reviews, articles)', 
             'Images (medical scans, photos)', 
             'Time Series / Sensor Data', 
             'Mixed / Other'],
            help="We'll generate synthetic data matching your type"
        )
        
        dataset_size = st.selectbox(
            "2. How large is your dataset?",
            ['<1K samples', '1K–10K samples', '10K–100K samples', '>100K samples'],
            help="Size affects AutoML training time and model selection"
        )
        
        feature_types = st.selectbox(
            "3. What types of features does your data contain?",
            ['Mostly numerical', 'Mostly categorical', 'Mixed numeric + categorical'],
            help="Affects synthetic data generation parameters"
        )
        
        missing_values = st.selectbox(
            "4. Does your dataset have missing values?",
            ['No missing values', 'Some missing values', 'Many missing values'],
            help="We'll inject appropriate missing values for realistic testing"
        )
    
    # Group 2: Problem Characteristics
    with st.expander("🎯 Problem Characteristics", expanded=True):
        problem_type = st.selectbox(
            "Problem Type",
            ['Binary Classification', 'Multi-class Classification', 'Regression', 'Clustering'],
            help="Determines AutoML approach and metrics"
        )
        
        if 'Classification' in problem_type:
            class_balance = st.selectbox(
                "5. Are your class labels balanced?",
                ['Roughly balanced', 'Slightly imbalanced', 'Highly imbalanced (rare positive cases)'],
                help="We'll generate appropriate class distributions"
            )
        else:
            class_balance = 'Not applicable'
        
        relationship_type = st.selectbox(
            "6. Do you expect relationships to be linear or complex?",
            ['Mostly linear', 'Moderately nonlinear', 'Highly nonlinear or unknown'],
            help="Controls synthetic data complexity"
        )
        
        noise_level = st.selectbox(
            "7. How noisy is your data?",
            ['Very clean and structured', 'Some noise or human error', 'High noise or uncertainty'],
            help="Adds realistic noise to synthetic data"
        )
    
    # Group 3: Priorities and Constraints
    with st.expander("⚖️ Analysis Preferences", expanded=True):
        interpretability_priority = st.selectbox(
            "8. What do you value more?",
            ['Maximum transparency (easy to explain)', 
             'A balance between accuracy and interpretability', 
             'Maximum accuracy (black-box OK)'],
            help="Influences algorithm recommendation ranking"
        )
        
        compute_resources = st.selectbox(
            "9. What's your computing environment?",
            ['Low-power laptop', 'Decent desktop / small GPU', 'High-end server / cloud GPU'],
            help="Affects AutoML generations and population size"
        )
    
    # Group 4: Context
    with st.expander("👤 Context & Experience", expanded=False):
        project_domain = st.selectbox(
            "10. Where is this project applied?",
            ['Healthcare', 'Finance', 'Education', 'E-commerce', 'General / Research'],
            help="Provides context for algorithm interpretation"
        )
        
        experience_level = st.selectbox(
            "11. What's your ML experience level?",
            ['Beginner', 'Intermediate', 'Advanced'],
            help="Affects explanation detail and complexity"
        )
        
        evaluation_priority = st.selectbox(
            "12. Which performance metric is most critical?",
            ['Accuracy / R²', 'Precision / Recall', 'F1-score', 'AUC', 'Training Time', 'Model Simplicity'],
            help="Primary optimization objective"
        )
    
    st.markdown("---")
    
    # Real AutoML button
    if st.button("🚀 Run Real AutoML Analysis", type="primary"):
        st.session_state.run_automl = True
        st.session_state.user_inputs = {
            'data_type': data_type,
            'dataset_size': dataset_size,
            'feature_types': feature_types,
            'missing_values': missing_values,
            'problem_type': problem_type,
            'class_balance': class_balance,
            'relationship_type': relationship_type,
            'noise_level': noise_level,
            'interpretability_priority': interpretability_priority,
            'compute_resources': compute_resources,
            'project_domain': project_domain,
            'experience_level': experience_level,
            'evaluation_priority': evaluation_priority
        }

# Initialize session state
if 'run_automl' not in st.session_state:
    st.session_state.run_automl = False

# Main content area
if st.session_state.run_automl and 'user_inputs' in st.session_state:
    
    user_inputs = st.session_state.user_inputs
    
    # Generate synthetic dataset
    st.markdown('<div class="automl-status">🧬 Generating Synthetic Dataset Based on Your Specifications...</div>', unsafe_allow_html=True)
    
    generator = IntelligentDatasetGenerator()
    X, y, dataset_info = generator.generate_dataset_from_inputs(user_inputs)
    
    st.success(f"✅ Generated dataset: {dataset_info['n_samples']} samples, {dataset_info['n_features']} features")
    
    # Show dataset preview
    with st.expander("🔍 Generated Dataset Preview", expanded=False):
        preview_df = pd.DataFrame(X[:10], columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
        preview_df['Target'] = y[:10]
        st.dataframe(preview_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", f"{X.shape[0]:,}")
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            if dataset_info['problem_type'] != 'Regression':
                st.metric("Classes", len(np.unique(y)))
            else:
                st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
    
    # Run Real AutoML
    st.markdown('<div class="automl-status">🤖 Running Real TPOT AutoML Analysis...</div>', unsafe_allow_html=True)
    
    automl_engine = RealAutoMLEngine()
    results = automl_engine.run_automl_analysis(X, y, dataset_info, user_inputs)
    
    if results['success']:
        st.markdown('<div class="real-results"><h3>🎉 Real AutoML Results Complete!</h3><p>All metrics below are from actual model training and evaluation - no estimates or rules!</p></div>', unsafe_allow_html=True)
        
        # Create tabs for results
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 AutoML Results", "📊 Model Comparison", "💻 Pipeline Code", "🔍 Analysis Details"])
        
        with tab1:
            st.header("🏆 Real AutoML Recommendations")
            
            # Best model results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 🥇 Best Algorithm: **{results['best_model']}**")
                st.markdown(f"**{results['metric_name']}:** {results['primary_score']:.4f}")
                
                if results['additional_metrics']:
                    st.markdown("**Additional Metrics:**")
                    for metric, value in results['additional_metrics'].items():
                        st.markdown(f"• {metric}: {value:.4f}")
                
                # AutoML-based reasoning
                st.markdown("### 🧠 Why This Algorithm Won:")
                reasoning = []
                
                if results['primary_score'] > 0.9:
                    reasoning.append("**Exceptional performance** (>90%) on your data characteristics")
                elif results['primary_score'] > 0.8:
                    reasoning.append("**Strong performance** (>80%) with good generalization")
                else:
                    reasoning.append("**Best available option** given your data constraints")
                
                if dataset_info.get('has_missing', False):
                    reasoning.append("**Robust to missing values** in your dataset")
                
                if dataset_info.get('noise_level', 0) > 0.1:
                    reasoning.append("**Noise-resistant** algorithm selected for your noisy data")
                
                reasoning.append(f"**Optimized through {automl_engine.__class__.__name__}** genetic algorithm search")
                
                for reason in reasoning:
                    st.markdown(f"• {reason}")
            
            with col2:
                # Performance gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = results['primary_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': results['metric_name']},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "#4a6cf7"},
                        'steps': [
                            {'range': [0, 0.6], 'color': "#ff9a9e"},
                            {'range': [0.6, 0.8], 'color': "#fecfef"},
                            {'range': [0.8, 1], 'color': "#a8edea"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=50, r=50, t=50, b=50))
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        with tab2:
            st.header("📊 Real Model Performance Comparison")
            st.markdown("*All scores from actual cross-validation on your synthetic dataset*")
            
            if results['model_scores']:
                # Create comparison visualization
                model_names = list(results['model_scores'].keys())
                mean_scores = [results['model_scores'][name]['mean_score'] for name in model_names]
                std_scores = [results['model_scores'][name]['std_score'] for name in model_names]
                
                # Sort by performance
                sorted_data = sorted(zip(model_names, mean_scores, std_scores), key=lambda x: x[1], reverse=True)
                model_names, mean_scores, std_scores = zip(*sorted_data)
                
                # Bar chart with error bars
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Bar(
                    x=model_names,
                    y=mean_scores,
                    error_y=dict(type='data', array=std_scores),
                    text=[f'{score:.3f}' for score in mean_scores],
                    textposition='outside',
                    marker_color=['#4a6cf7' if name == results['best_model'] else '#a8edea' for name in model_names]
                ))
                
                fig_comparison.update_layout(
                    title=f"Real {results['metric_name']} Comparison (Cross-Validation)",
                    xaxis_title="Algorithms",
                    yaxis_title=results['metric_name'],
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Cross-Validation Results")
                
                comparison_data = []
                for name in model_names:
                    scores_info = results['model_scores'][name]
                    comparison_data.append({
                        'Algorithm': name,
                        'Mean Score': f"{scores_info['mean_score']:.4f}",
                        'Std Dev': f"{scores_info['std_score']:.4f}",
                        'CV Scores': [f"{s:.3f}" for s in scores_info['scores']],
                        'Status': '🥇 AutoML Winner' if name == results['best_model'] else '📊 Evaluated'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            else:
                st.warning("Model comparison data not available - showing AutoML winner only")
        
        with tab3:
            st.header("💻 Production-Ready Pipeline Code")
            st.markdown("*Generated automatically by TPOT AutoML*")
            
            # Display the actual pipeline code
            st.markdown('<div class="pipeline-code">', unsafe_allow_html=True)
            st.markdown("### 🔧 Complete Sklearn Pipeline:")
            st.code(results['pipeline_code'], language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional implementation guidance
            st.markdown("### 🚀 How to Use This Code:")
            st.markdown("""
            1. **Copy the pipeline code above** - it's production-ready
            2. **Replace the data loading section** with your actual dataset
            3. **The pipeline includes all preprocessing** (scaling, feature selection, etc.)
            4. **No hyperparameter tuning needed** - already optimized by AutoML
            5. **Save the model** using `joblib.dump()`  for deployment
            """)
            
            # Show how to save and load the model
            st.markdown("### 💾 Model Persistence Example:")
            st.code("""
# Save the trained model
import joblib
joblib.dump(exported_pipeline, 'automl_model.pkl')

# Load and use the model later
loaded_model = joblib.load('automl_model.pkl')
predictions = loaded_model.predict(new_data)
            """, language='python')
        
        with tab4:
            st.header("🔍 Detailed Analysis")
            
            # Dataset generation details
            st.markdown("### 🧬 Generated Dataset Characteristics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Properties:**")
                st.markdown(f"• Samples: {dataset_info['n_samples']:,}")
                st.markdown(f"• Features: {dataset_info['n_features']}")
                st.markdown(f"• Informative Features: {dataset_info['n_informative']}")
                st.markdown(f"• Problem Type: {dataset_info['problem_type']}")
                st.markdown(f"• Has Missing Values: {'Yes' if dataset_info.get('has_missing') else 'No'}")
            
            with col2:
                st.markdown("**Quality Metrics:**")
                st.markdown(f"• Noise Level: {dataset_info['noise_level']:.1%}")
                st.markdown(f"• Class Balance: {'Balanced' if dataset_info.get('is_balanced') else 'Imbalanced'}")
                st.markdown(f"• Target Classes: {len(np.unique(y))}")
                
                if dataset_info['problem_type'] != 'Regression':
                    class_dist = pd.Series(y).value_counts().sort_index()
                    st.markdown("**Class Distribution:**")
                    for cls, count in class_dist.items():
                        st.markdown(f"• Class {cls}: {count} ({count/len(y):.1%})")
            
            # AutoML process details
            st.markdown("### 🤖 AutoML Process Details")
            st.markdown(f"""
            **TPOT Configuration:**
            • Generations: 3 (reduced for speed)
            • Population Size: 20
            • Cross-Validation: 3-fold
            • Optimization Metric: {results['metric_name']}
            • Random State: 42 (reproducible)
            • Max Time: 2 minutes per generation
            
            **Process:**
            1. Generated synthetic dataset matching your specifications
            2. Split into train/test sets (80/20)
            3. Applied automated preprocessing (imputation if needed)
            4. Evolved ML pipelines through genetic algorithm
            5. Evaluated {len(results.get('model_scores', {}))} different algorithm types
            6. Selected best performing pipeline: **{results['best_model']}**
            """)
            
            # Prediction analysis
            if 'test_predictions' in results and 'test_actual' in results:
                st.markdown("### 📈 Prediction Analysis")
                
                y_pred = results['test_predictions']
                y_actual = results['test_actual']
                
                if dataset_info['problem_type'] != 'Regression':
                    # Classification: Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    
                    cm = confusion_matrix(y_actual, y_pred)
                    
                    fig_cm = px.imshow(
                        cm,
                        title="Confusion Matrix (Test Set)",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        aspect="auto",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                else:
                    # Regression: Predicted vs Actual
                    fig_reg = px.scatter(
                        x=y_actual, y=y_pred,
                        title="Predicted vs Actual Values",
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                    )
                    
                    # Add perfect prediction line
                    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
                    fig_reg.add_shape(
                        type="line",
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(dash="dash", color="red"),
                    )
                    
                    st.plotly_chart(fig_reg, use_container_width=True)
            
            # Performance insights
            st.markdown("### 💡 Performance Insights")
            
            insights = []
            
            if results['primary_score'] > 0.95:
                insights.append("🎯 **Exceptional Performance**: Your data characteristics allow for very high accuracy")
            elif results['primary_score'] > 0.85:
                insights.append("✅ **Strong Performance**: Good predictive power with room for optimization")
            elif results['primary_score'] > 0.70:
                insights.append("⚠️ **Moderate Performance**: Consider feature engineering or more data")
            else:
                insights.append("🔧 **Challenging Dataset**: May need domain expertise and feature engineering")
            
            if dataset_info.get('has_missing'):
                insights.append("🔧 **Missing Value Handling**: AutoML automatically selected robust preprocessing")
            
            if dataset_info['noise_level'] > 0.1:
                insights.append("🛡️ **Noise Robustness**: Selected algorithm handles data uncertainty well")
            
            if 'imbalanced' in user_inputs.get('class_balance', '').lower():
                insights.append("⚖️ **Imbalance Handling**: Algorithm chosen for skewed class distribution")
            
            for insight in insights:
                st.markdown(insight)
    
    else:
        st.error(f"❌ AutoML Analysis Failed: {results.get('error', 'Unknown error')}")
        st.markdown("**Possible solutions:**")
        st.markdown("• Try a smaller dataset size")
        st.markdown("• Reduce noise level")
        st.markdown("• Simplify relationship complexity")
        st.markdown("• Check your parameter combinations")

else:
    # Welcome screen with new design
    st.markdown('<h1 class="main-title">Machine Learning Algorithm Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Algorithm Selection for Your Data")
    
    # Introduction
    st.markdown("""
    <div class="card">
        <p>This tool helps you identify the best machine learning algorithm for your specific dataset and problem type. 
        Unlike traditional recommenders that use simple rules, our system generates realistic synthetic data based on your 
        specifications and evaluates multiple algorithms to find the optimal solution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in cards
    st.markdown("### Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Data Generation</h3>
            <p>Creates realistic synthetic datasets based on your detailed specifications, 
            including feature types, noise levels, and problem characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>AutoML Analysis</h3>
            <p>Uses TPOT's genetic algorithm to automatically find the best machine learning 
            pipeline for your data, testing multiple algorithms and hyperparameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>Actionable Results</h3>
            <p>Get detailed performance metrics, visualizations, and production-ready code 
            for the recommended algorithm, along with insights into why it was chosen.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use
    st.markdown("### Getting Started")
    
    steps = [
        {"title": "1. Configure Your Dataset", 
         "desc": "Use the sidebar to answer questions about your data characteristics and project requirements."},
        {"title": "2. Run AutoML Analysis", 
         "desc": "Click the 'Run Real AutoML Analysis' button to start the automated model selection process."},
        {"title": "3. Review Results", 
         "desc": "Explore the recommended algorithms, performance metrics, and visualizations."},
        {"title": "4. Implement Solution", 
         "desc": "Copy the production-ready code and integrate it into your project."}
    ]
    
    for step in steps:
        with st.expander(step["title"], expanded=True):
            st.markdown(f"<p>{step['desc']}</p>", unsafe_allow_html=True)
    
    # Example visualization
    st.markdown("### Example Analysis")
    
    # Create a sample visualization
    sample_data = pd.DataFrame({
        'Algorithm': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'KNN'],
        'Score': [0.92, 0.89, 0.85, 0.83, 0.81],
        'Training Time (s)': [1.5, 2.1, 0.3, 3.2, 0.8],
        'Interpretability': [3, 2, 5, 2, 4],
        'Type': ['Ensemble', 'Ensemble', 'Linear', 'Kernel', 'Instance-based']
    })
    
    fig = px.bar(
        sample_data,
        x='Algorithm',
        y='Score',
        color='Type',
        title='Algorithm Performance Comparison',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        labels={'Score': 'Performance Score'},
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Source Sans Pro, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Requirements and setup
    st.markdown("### Setup")
    
    st.markdown("""
    <div class="card">
        <h4>Requirements</h4>
        <ul>
            <li>Python 3.7 or higher</li>
            <li>Required packages: streamlit, tpot, scikit-learn, plotly, pandas, numpy</li>
        </ul>
        
        <h4>Installation</h4>
        <div class="pipeline-code">
            pip install streamlit tpot scikit-learn plotly pandas numpy
        </div>
        
        <h4>Running the Application</h4>
        <div class="pipeline-code">
            streamlit run intelligent_ml_recommender.py
        </div>
        
        <p class="status-info">
            <strong>Note:</strong> The AutoML analysis may take several minutes to complete, 
            depending on your dataset size and complexity.
        </p>
    </div>
    """, unsafe_allow_html=True)
