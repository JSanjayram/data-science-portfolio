import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(df.head())
print( df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
print(df.describe())

df['Attrition'].value_counts()  # Check the distribution of the 'Attrition' target

# Remove duplicates if any
df = df.drop_duplicates()

# Label Encoding for the target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop non-predictive columns
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')

# Check for remaining string columns
print("\nData types after initial processing:")
print(df.dtypes)
print("\nString columns remaining:")
string_cols = df.select_dtypes(include=['object']).columns.tolist()
print(string_cols)

# Handle all categorical columns
categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
existing_cats = [col for col in categorical_cols if col in df.columns]

# One-hot encoding for categorical columns
if existing_cats:
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

# Handle any remaining object columns
remaining_objects = df.select_dtypes(include=['object']).columns.tolist()
if remaining_objects:
    print(f"\nEncoding remaining object columns: {remaining_objects}")
    for col in remaining_objects:
        print(f"Unique values in {col}: {df[col].unique()}")
    df = pd.get_dummies(df, columns=remaining_objects, drop_first=True)

# Get numerical columns (excluding target)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Attrition' in numerical_cols:
    numerical_cols.remove('Attrition')

print(f"\nFinal numerical columns: {len(numerical_cols)}")
print(f"Final data shape: {df.shape}")

# Scaling the numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Final check for data types
print("\nFinal data types:")
print(df.dtypes.value_counts())

x = df.drop('Attrition', axis=1)
y = df['Attrition']

# Ensure all features are numeric
print(f"\nFeature matrix shape: {x.shape}")
print(f"Non-numeric columns in X: {x.select_dtypes(exclude=[np.number]).columns.tolist()}")

# Convert to numpy arrays to ensure compatibility
x = x.astype(float)
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {x_train.shape}, Test set size: {x_test.shape}")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

import matplotlib.pyplot as plt
# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Best hyperparameters
print("Best Parameters:", grid_search.best_params_)

# Feature Importance from Random Forest
feature_importance = rf.feature_importances_
sorted_idx = feature_importance.argsort()

# Plot top 10 features
top_features = sorted_idx[-10:]
plt.figure(figsize=(10, 6))
plt.barh(x_train.columns[top_features], feature_importance[top_features])
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
