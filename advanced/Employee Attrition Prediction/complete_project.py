import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EMPLOYEE ATTRITION PREDICTION - COMPLETE PROJECT")
print("="*60)

# STEP 1: DATA UNDERSTANDING
print("\n1. DATA UNDERSTANDING")
print("-" * 30)
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nDataset info:")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")
print(f"\nClass distribution:")
print(df['Attrition'].value_counts())
print(f"\nClass distribution (%):")
print(df['Attrition'].value_counts(normalize=True) * 100)

# STEP 2: DATA CLEANING
print("\n2. DATA CLEANING")
print("-" * 30)
print(f"Duplicates found: {df.duplicated().sum()}")
df = df.drop_duplicates()
print("Duplicates removed")

# Remove non-predictive columns
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')
print("Non-predictive columns removed")

# Handle inconsistent entries
print(f"Unique values in Over18: {df['Over18'].unique()}")
df = df.drop(['Over18'], axis=1)  # All values are 'Y'
print("Over18 column removed (constant value)")

# STEP 3: FEATURE ENGINEERING
print("\n3. FEATURE ENGINEERING")
print("-" * 30)
# Create new features
df['YearsSinceLastPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['OverTime_Hours'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['TotalSatisfaction'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                          df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
df['IncomePerYear'] = df['MonthlyIncome'] * 12
df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

print("New features created:")
print("- YearsSinceLastPromotion (adjusted)")
print("- OverTime_Hours")
print("- TotalSatisfaction")
print("- IncomePerYear")
print("- ExperienceRatio")

# STEP 4: ENCODING
print("\n4. ENCODING")
print("-" * 30)
# Encode target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print("Target variable encoded: Yes=1, No=0")

# One-hot encode categorical variables
categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
print(f"Encoding categorical columns: {categorical_cols}")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Dataset shape after encoding: {df.shape}")

# STEP 5: FEATURE SCALING
print("\n5. FEATURE SCALING")
print("-" * 30)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Identify numerical columns for scaling
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns to scale: {len(numerical_cols)}")

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("Feature scaling completed using StandardScaler")

# STEP 6: MODEL BUILDING
print("\n6. MODEL BUILDING")
print("-" * 30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

model_results = {}
print("\nTraining models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    model_results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    print(f"{name} trained successfully")

# STEP 7: MODEL EVALUATION
print("\n7. MODEL EVALUATION")
print("-" * 30)
evaluation_results = {}

for name, result in model_results.items():
    y_pred = result['predictions']
    y_pred_proba = result['probabilities']
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    
    evaluation_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Create evaluation comparison
eval_df = pd.DataFrame(evaluation_results).T
print(f"\nModel Comparison:")
print(eval_df.round(4))

# STEP 8: HYPERPARAMETER TUNING
print("\n8. HYPERPARAMETER TUNING")
print("-" * 30)
# Find best performing model for tuning
best_model_name = eval_df['F1-Score'].idxmax()
print(f"Best model for tuning: {best_model_name}")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    base_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    base_model = LogisticRegression(max_iter=1000, random_state=42)
else:  # Decision Tree
    param_grid = {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base_model = DecisionTreeClassifier(random_state=42)

print("Performing GridSearchCV...")
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

# Evaluate tuned model
best_tuned_model = grid_search.best_estimator_
y_pred_tuned = best_tuned_model.predict(X_test)
tuned_f1 = f1_score(y_test, y_pred_tuned)
print(f"Tuned model test F1-score: {tuned_f1:.4f}")

# STEP 9: MODEL INTERPRETATION
print("\n9. MODEL INTERPRETATION")
print("-" * 30)
# Use the best model for interpretation
if hasattr(best_tuned_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_tuned_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_final.csv', index=False)

# STEP 10: SAVE RESULTS
print("\n10. SAVING RESULTS")
print("-" * 30)
# Save cleaned dataset
df.to_csv('cleaned_employee_attrition_final.csv', index=False)

# Save model evaluation results
eval_df.to_csv('model_evaluation_results.csv')

# Save best model parameters
with open('best_model_params.txt', 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best CV F1-Score: {grid_search.best_score_:.4f}\n")
    f.write(f"Test F1-Score: {tuned_f1:.4f}\n")

print("Files saved:")
print("- cleaned_employee_attrition_final.csv")
print("- feature_importance_final.csv")
print("- feature_importance_final.png")
print("- model_evaluation_results.csv")
print("- best_model_params.txt")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)