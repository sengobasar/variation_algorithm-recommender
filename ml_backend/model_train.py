import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load dataset
data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

X = dataset.copy()
y = data['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Train model
clf = DecisionTreeClassifier(max_depth=4, ccp_alpha=0.01, random_state=42)
clf.fit(X_train, y_train)

# 4. Save model and features
joblib.dump(clf, "breast_cancer_tree.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("✅ Model and features saved successfully.")
