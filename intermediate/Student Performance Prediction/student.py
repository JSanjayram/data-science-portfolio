import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle

# Step 1: Data Understanding
df = pd.read_csv('StudentPerformanceFactors.csv')
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Step 2: Data Cleaning
# Handle missing values properly
numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                   'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                   'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 
                   'Distance_from_Home', 'Gender']

# Fill missing values
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

# Remove duplicates and invalid data
df = df.drop_duplicates().copy()
df = df[(df['Hours_Studied'] >= 0) & (df['Attendance'] >= 0)].copy()

print(f"\nAfter cleaning - Shape: {df.shape}")

# Step 3: Outlier Detection and Removal
outlier_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']

plt.figure(figsize=(12, 6))
for i, col in enumerate(outlier_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Before: {col}')
plt.tight_layout()
plt.show()

# Remove outliers using IQR
Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = ~((df[outlier_cols] < lower_bound) | (df[outlier_cols] > upper_bound)).any(axis=1)
df_cleaned = df[mask].copy()
print(f"Outliers removed: {df.shape[0] - df_cleaned.shape[0]} rows")

# Step 4: Feature Engineering
df_cleaned.loc[:, 'Study_Efficiency'] = df_cleaned['Previous_Scores'] / (df_cleaned['Hours_Studied'] + 1)

# Map education levels to numbers
education_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_cleaned.loc[:, 'Parental_Education_Num'] = df_cleaned['Parental_Education_Level'].map(education_map)
df_cleaned['Parental_Education_Num'] = df_cleaned['Parental_Education_Num'].fillna(2)  # Default to Medium

# Create parent support score
involvement_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_cleaned.loc[:, 'Parental_Involvement_Num'] = df_cleaned['Parental_Involvement'].map(involvement_map)
df_cleaned['Parental_Involvement_Num'] = df_cleaned['Parental_Involvement_Num'].fillna(2)

df_cleaned.loc[:, 'Parent_Support_Score'] = df_cleaned['Parental_Involvement_Num'] * df_cleaned['Parental_Education_Num']

# Step 5: Encoding
le = LabelEncoder()
df_cleaned.loc[:, 'Gender_Encoded'] = le.fit_transform(df_cleaned['Gender'].astype(str))

# One-hot encoding for categorical variables
categorical_to_encode = ['Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 
                        'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 
                        'Peer_Influence', 'Learning_Disabilities', 'Distance_from_Home']

for col in categorical_to_encode:
    if col in df_cleaned.columns:
        df_cleaned = pd.get_dummies(df_cleaned, columns=[col], prefix=col, drop_first=True)

# Step 6: Feature Scaling
scaler = StandardScaler()
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                     'Study_Efficiency', 'Parent_Support_Score', 'Tutoring_Sessions', 'Physical_Activity']

# Only scale existing numerical columns
existing_numerical = [col for col in numerical_features if col in df_cleaned.columns]
df_cleaned[existing_numerical] = scaler.fit_transform(df_cleaned[existing_numerical])

# Step 7: Create Target Variable and Prepare Data
threshold = 60
df_cleaned['Pass_Fail'] = (df_cleaned['Exam_Score'] >= threshold).astype(int)

# Prepare features and target
X = df_cleaned.drop(['Exam_Score', 'Pass_Fail', 'Parental_Education_Level', 'Parental_Involvement', 
                     'Gender', 'Parental_Education_Num', 'Parental_Involvement_Num'], axis=1, errors='ignore')

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])
y = df_cleaned['Pass_Fail']

# Handle any remaining NaN values
X = X.fillna(0)

print(f"Final feature shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Building and Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name} Performance:")
    for metric, score in results[name].items():
        print(f"{metric}: {score:.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Step 9: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Use compatible RandomForest without problematic parameters
base_rf = RandomForestClassifier(random_state=42, n_jobs=1)
grid_search = GridSearchCV(base_rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_pred = best_rf.predict(X_test)
best_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Model Accuracy: {accuracy_score(y_test, best_pred):.3f}")
print(f"Best Model ROC-AUC: {roc_auc_score(y_test, best_pred_proba):.3f}")

# Step 10: Feature Importance
importances = best_rf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save models and preprocessors
with open('best_student_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('student_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('student_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModels saved successfully!")
print("- best_student_model.pkl")
print("- student_scaler.pkl") 
print("- student_features.pkl")