import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Dataset shape: {df.shape}")
print(f"Attrition distribution:\n{df['Attrition'].value_counts(normalize=True) * 100}")

# Data cleaning
df = df.drop_duplicates()
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')

# Handle categorical variables properly
categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

# Check for any remaining object columns
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"Remaining object columns: {object_cols}")
    for col in object_cols:
        print(f"Unique values in {col}: {df[col].unique()}")
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

# Prepare features
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Ensure all data is numeric
X = X.astype(float)
y = y.astype(int)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy}
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Feature importance
best_model = results['Random Forest (Balanced)']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize top features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Satisfaction analysis
original_df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
original_df['Attrition'] = original_df['Attrition'].map({'Yes': 1, 'No': 0})
satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
corr_data = original_df[satisfaction_cols + ['Attrition']].corr()['Attrition'].drop('Attrition')

plt.figure(figsize=(10, 6))
corr_data.plot(kind='bar')
plt.title('Correlation between Satisfaction Metrics and Attrition')
plt.ylabel('Correlation with Attrition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('satisfaction_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
df.to_csv('cleaned_employee_attrition.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))
print(f"\nSatisfaction Correlations:")
print(corr_data.sort_values())
print(f"\nFiles saved:")
print("- cleaned_employee_attrition.csv")
print("- feature_importance.csv")
print("- feature_importance.png")
print("- satisfaction_correlation.png")