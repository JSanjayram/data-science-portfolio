import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Dataset shape: {df.shape}")

# Check class imbalance
print(f"\nAttrition distribution:")
print(df['Attrition'].value_counts(normalize=True) * 100)

# Data cleaning
df = df.drop_duplicates()
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')

# One-hot encoding
categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare features
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale features
numerical_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# Train and evaluate models
for name, model in models.items():
    if 'SMOTE' in name:
        model.fit(X_train_balanced, y_train_balanced)
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy}
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Feature importance analysis
best_model = results['Random Forest (SMOTE)']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance - Random Forest with SMOTE')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Satisfaction correlation analysis
satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
original_df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
original_df['Attrition'] = original_df['Attrition'].map({'Yes': 1, 'No': 0})
corr_data = original_df[satisfaction_cols + ['Attrition']].corr()['Attrition'].drop('Attrition')

plt.figure(figsize=(10, 6))
corr_data.plot(kind='bar')
plt.title('Correlation between Satisfaction Metrics and Attrition')
plt.ylabel('Correlation with Attrition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save cleaned dataset and results
df.to_csv('cleaned_employee_attrition.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))
print(f"\nFiles saved:")
print("- cleaned_employee_attrition.csv")
print("- feature_importance.csv")