"""
Algorithm Recommender for Machine Learning Projects

This module provides algorithm recommendations based on project specifications.
"""
from typing import List, Dict, Any
import streamlit as st

class AlgorithmRecommender:
    """Recommends ML algorithms based on project specifications."""
    
    @staticmethod
    def recommend_algorithms(problem_type: str = 'binary classification',
                          dataset_size: str = 'medium',
                          feature_types: str = 'mixed',
                          missing_values: str = 'yes, few',
                          class_balance: str = 'balanced',
                          relationship_type: str = 'moderately nonlinear',
                          noise_level: str = 'medium',
                          priority: str = 'balanced accuracy & interpretability',
                          computing_env: str = 'medium desktop',
                          project_domain: str = 'general',
                          experience_level: str = 'intermediate',
                          metric: str = 'accuracy') -> Dict[str, List[str]]:
        """
        Recommend ML algorithms based on project specifications.
        
        Args:
            problem_type: Type of ML problem (binary/multi-class classification, regression, clustering)
            dataset_size: Size of the dataset (small, medium, large)
            feature_types: Type of features (numerical, categorical, mixed)
            missing_values: Presence of missing values (yes/no, few/many)
            class_balance: Class distribution (balanced, slightly imbalanced, highly imbalanced)
            relationship_type: Nature of relationships in data (linear, moderately nonlinear, highly nonlinear)
            noise_level: Amount of noise in the data (low, medium, high)
            priority: Project priorities (accuracy, interpretability, speed, etc.)
            computing_env: Available computing resources (laptop, medium desktop, cloud, etc.)
            project_domain: Domain of the project (e.g., healthcare, finance, e-commerce)
            experience_level: User's ML experience level (beginner, intermediate, expert)
            metric: Primary evaluation metric (accuracy, precision, recall, f1, etc.)
            
        Returns:
            Dict with 'preprocessing' and 'algorithms' keys containing lists of recommended preprocessing steps and algorithm class names
        """
        # Determine preprocessing steps
        preprocessing = []
        if 'yes' in missing_values:
            preprocessing.append('SimpleImputer')
        if feature_types in ['mixed', 'mostly categorical']:
            preprocessing.append('OneHotEncoder')
        if feature_types in ['mixed', 'mostly numerical']:
            preprocessing.append('StandardScaler')
        if class_balance == 'highly imbalanced':
            preprocessing.append('SMOTE')
        if noise_level == 'high':
            preprocessing.append('RobustScaler')
        if relationship_type == 'highly nonlinear' and priority == 'maximum accuracy':
            preprocessing.append('PolynomialFeatures')
        
        # Determine algorithms
        algorithms = []
        
        # For the specific requirements of telecom churn prediction
        if (problem_type == 'binary classification' and 
            dataset_size == 'medium' and 
            class_balance == 'highly imbalanced' and 
            metric.lower() == 'f1-score' and
            project_domain.lower() == 'e-commerce'):
            algorithms = [
                'RandomForestClassifier',
                'XGBClassifier',
                'LGBMClassifier',
                'LogisticRegression',
                'GradientBoostingClassifier'
            ]
            
        # Default recommendations for binary classification
        elif problem_type.lower() in ['binary classification', 'binary']:
            algorithms = [
                'RandomForestClassifier',
                'LogisticRegression',
                'GradientBoostingClassifier',
                'XGBClassifier',
                'LGBMClassifier'
            ]
            
        # For regression problems
        elif problem_type.lower() == 'regression':
            algorithms = [
                'RandomForestRegressor',
                'XGBRegressor',
                'GradientBoostingRegressor',
                'LinearRegression',
                'LGBMRegressor'
            ]
            
        # For multi-class classification
        elif problem_type.lower() in ['multiclass classification', 'multiclass']:
            algorithms = [
                'RandomForestClassifier',
                'XGBClassifier',
                'LGBMClassifier',
                'LogisticRegression',
                'GradientBoostingClassifier'
            ]
            
        # For clustering
        elif problem_type.lower() == 'clustering':
            algorithms = [
                'KMeans',
                'DBSCAN',
                'AgglomerativeClustering',
                'GaussianMixture',
                'SpectralClustering'
            ]
            
        # Default fallback
        else:
            algorithms = [
                'RandomForestClassifier',
                'XGBClassifier',
                'LogisticRegression',
                'GradientBoostingClassifier',
                'SVC'
            ]
        
        return {'preprocessing': preprocessing, 'algorithms': algorithms}


def get_algorithm_recommendations() -> Dict[str, List[str]]:
    """Get algorithm recommendations for telecom churn prediction."""
    recommender = AlgorithmRecommender()
    return recommender.recommend_algorithms(
        problem_type='binary classification',
        dataset_size='medium',
        feature_types='mixed',
        missing_values='yes, few',
        class_balance='highly imbalanced',
        relationship_type='moderately nonlinear',
        noise_level='medium',
        priority='balanced accuracy & interpretability',
        computing_env='medium desktop',
        project_domain='e-commerce',
        experience_level='intermediate',
        metric='F1-score'
    )


if __name__ == "__main__":
    st.title("ML Algorithm Recommender")
    
    # Input widgets for parameters
    problem_type = st.selectbox("Problem Type", ['binary classification', 'multi-class classification', 'regression', 'clustering'])
    dataset_size = st.selectbox("Dataset Size", ['small', 'medium', 'large'])
    feature_types = st.selectbox("Feature Types", ['mostly numerical', 'mostly categorical', 'mixed'])
    missing_values = st.selectbox("Missing Values", ['yes, few', 'yes, many', 'no'])
    class_balance = st.selectbox("Class Balance", ['balanced', 'slightly imbalanced', 'highly imbalanced'])
    relationship_type = st.selectbox("Relationship Type", ['linear', 'moderately nonlinear', 'highly nonlinear'])
    noise_level = st.selectbox("Noise Level", ['low', 'medium', 'high'])
    priority = st.selectbox("Priority", ['maximum transparency', 'balanced accuracy & interpretability', 'maximum accuracy'])
    computing_env = st.selectbox("Computing Environment", ['low-power laptop', 'medium desktop', 'high-end server'])
    project_domain = st.selectbox("Project Domain", ['healthcare', 'finance', 'e-commerce', 'education', 'general research'])
    experience_level = st.selectbox("ML Experience Level", ['beginner', 'intermediate', 'advanced'])
    metric = st.selectbox("Critical Performance Metric", ['accuracy', 'precision', 'recall', 'F1-score', 'AUC', 'training time', 'model simplicity'])
    
    if st.button("Run Analysis"):
        # Get recommendations
        recommender = AlgorithmRecommender()
        recommendations = recommender.recommend_algorithms(
            problem_type=problem_type,
            dataset_size=dataset_size,
            feature_types=feature_types,
            missing_values=missing_values,
            class_balance=class_balance,
            relationship_type=relationship_type,
            noise_level=noise_level,
            priority=priority,
            computing_env=computing_env,
            project_domain=project_domain,
            experience_level=experience_level,
            metric=metric
        )
        
        # Display recommendations
        st.subheader("Recommended Preprocessing Steps:")
        if recommendations['preprocessing']:
            st.markdown("\n".join(f"- {step}" for step in recommendations['preprocessing']))
        else:
            st.write("No specific preprocessing recommended.")
        
        st.subheader("Recommended Algorithms:")
        st.markdown("\n".join(f"- {algo}" for algo in recommendations['algorithms']))
