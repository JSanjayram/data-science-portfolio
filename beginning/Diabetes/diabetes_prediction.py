import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Generate diabetes dataset
def create_diabetes_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate correlated features
    age = np.random.normal(45, 15, n_samples).clip(18, 80)
    bmi = np.random.normal(28, 6, n_samples).clip(15, 50)
    glucose = np.random.normal(120, 30, n_samples).clip(70, 200)
    blood_pressure = np.random.normal(80, 15, n_samples).clip(60, 120)
    insulin = np.random.normal(100, 50, n_samples).clip(0, 300)
    
    # Create diabetes probability based on risk factors
    diabetes_prob = (
        (age - 18) / 62 * 0.3 +
        (bmi - 15) / 35 * 0.4 +
        (glucose - 70) / 130 * 0.5 +
        (blood_pressure - 60) / 60 * 0.2 +
        (insulin - 0) / 300 * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)
    
    # Generate binary diabetes outcome
    diabetes = (diabetes_prob > 0.5).astype(int)
    
    # Additional features
    pregnancies = np.random.poisson(2, n_samples).clip(0, 10)
    skin_thickness = np.random.normal(25, 8, n_samples).clip(0, 50)
    
    data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'age': age,
        'diabetes': diabetes
    }
    
    return pd.DataFrame(data)

# Exploratory Data Analysis
def perform_eda(df):
    plt.figure(figsize=(16, 12))
    
    # 1. Target distribution
    plt.subplot(3, 4, 1)
    df['diabetes'].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title('Diabetes Distribution')
    plt.xlabel('Diabetes (0=No, 1=Yes)')
    
    # 2. Age distribution by diabetes
    plt.subplot(3, 4, 2)
    df.boxplot(column='age', by='diabetes', ax=plt.gca())
    plt.title('Age Distribution by Diabetes')
    plt.suptitle('')
    
    # 3. BMI distribution by diabetes
    plt.subplot(3, 4, 3)
    df.boxplot(column='bmi', by='diabetes', ax=plt.gca())
    plt.title('BMI Distribution by Diabetes')
    plt.suptitle('')
    
    # 4. Glucose distribution by diabetes
    plt.subplot(3, 4, 4)
    df.boxplot(column='glucose', by='diabetes', ax=plt.gca())
    plt.title('Glucose Distribution by Diabetes')
    plt.suptitle('')
    
    # 5. Correlation heatmap
    plt.subplot(3, 4, 5)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    # 6. Glucose vs BMI scatter
    plt.subplot(3, 4, 6)
    colors = ['blue', 'red']
    for i, diabetes_status in enumerate([0, 1]):
        subset = df[df['diabetes'] == diabetes_status]
        plt.scatter(subset['glucose'], subset['bmi'], 
                   c=colors[i], alpha=0.6, 
                   label=f'Diabetes: {diabetes_status}')
    plt.xlabel('Glucose Level')
    plt.ylabel('BMI')
    plt.title('Glucose vs BMI by Diabetes')
    plt.legend()
    
    # 7. Age histogram
    plt.subplot(3, 4, 7)
    df[df['diabetes']==0]['age'].hist(alpha=0.7, label='No Diabetes', bins=20)
    df[df['diabetes']==1]['age'].hist(alpha=0.7, label='Diabetes', bins=20)
    plt.xlabel('Age')
    plt.title('Age Distribution')
    plt.legend()
    
    # 8. BMI histogram
    plt.subplot(3, 4, 8)
    df[df['diabetes']==0]['bmi'].hist(alpha=0.7, label='No Diabetes', bins=20)
    df[df['diabetes']==1]['bmi'].hist(alpha=0.7, label='Diabetes', bins=20)
    plt.xlabel('BMI')
    plt.title('BMI Distribution')
    plt.legend()
    
    # 9. Blood pressure analysis
    plt.subplot(3, 4, 9)
    df.boxplot(column='blood_pressure', by='diabetes', ax=plt.gca())
    plt.title('Blood Pressure by Diabetes')
    plt.suptitle('')
    
    # 10. Insulin levels
    plt.subplot(3, 4, 10)
    df.boxplot(column='insulin', by='diabetes', ax=plt.gca())
    plt.title('Insulin Levels by Diabetes')
    plt.suptitle('')
    
    # 11. Pregnancies impact
    plt.subplot(3, 4, 11)
    pregnancy_diabetes = df.groupby('pregnancies')['diabetes'].mean()
    pregnancy_diabetes.plot(kind='bar')
    plt.title('Diabetes Rate by Pregnancies')
    plt.xlabel('Number of Pregnancies')
    plt.ylabel('Diabetes Rate')
    
    # 12. Feature importance preview
    plt.subplot(3, 4, 12)
    feature_cols = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'age']
    diabetes_corr = df[feature_cols + ['diabetes']].corr()['diabetes'].drop('diabetes')
    diabetes_corr.abs().sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Correlation with Diabetes')
    
    plt.tight_layout()
    plt.savefig('diabetes_eda.png', dpi=300, bbox_inches='tight')
    plt.show()

# Train logistic regression model
def train_logistic_regression(df):
    # Prepare features and target
    feature_cols = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'age']
    X = df[feature_cols]
    y = df['diabetes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return lr_model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba, feature_cols

# Visualize model results
def visualize_model_results(lr_model, X_test_scaled, y_test, y_pred, y_pred_proba, feature_cols):
    plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC Curve
    plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 3. Feature Coefficients
    plt.subplot(2, 3, 3)
    coefficients = lr_model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefficients})
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    
    # 4. Prediction Probability Distribution
    plt.subplot(2, 3, 4)
    plt.hist(y_pred_proba[y_test==0], alpha=0.7, label='No Diabetes', bins=20)
    plt.hist(y_pred_proba[y_test==1], alpha=0.7, label='Diabetes', bins=20)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    # 5. Feature Importance (absolute coefficients)
    plt.subplot(2, 3, 5)
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': np.abs(coefficients)})
    importance_df = importance_df.sort_values('Importance', ascending=True)
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance (|Coefficients|)')
    plt.xlabel('Absolute Coefficient Value')
    
    # 6. Prediction vs Actual
    plt.subplot(2, 3, 6)
    plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='Actual', s=20)
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.6, label='Predicted', s=20)
    plt.xlabel('Sample Index')
    plt.ylabel('Diabetes (0/1)')
    plt.title('Actual vs Predicted')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('diabetes_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate insights
def generate_insights(df, lr_model, feature_cols, accuracy, auc_score):
    insights = []
    
    # Model performance
    insights.append(f"Logistic Regression Accuracy: {accuracy:.4f}")
    insights.append(f"AUC Score: {auc_score:.4f} ({'Excellent' if auc_score > 0.8 else 'Good' if auc_score > 0.7 else 'Fair'})")
    
    # Feature importance
    coefficients = lr_model.coef_[0]
    most_important_idx = np.argmax(np.abs(coefficients))
    most_important_feature = feature_cols[most_important_idx]
    insights.append(f"Most important feature: {most_important_feature}")
    
    # Dataset insights
    diabetes_rate = df['diabetes'].mean()
    insights.append(f"Overall diabetes rate in dataset: {diabetes_rate:.3f}")
    
    # Risk factors
    high_risk_bmi = df[df['bmi'] > 30]['diabetes'].mean()
    high_risk_glucose = df[df['glucose'] > 140]['diabetes'].mean()
    insights.append(f"Diabetes rate with BMI > 30: {high_risk_bmi:.3f}")
    insights.append(f"Diabetes rate with glucose > 140: {high_risk_glucose:.3f}")
    
    return insights

def main():
    print("🩺 Diabetes Prediction using Logistic Regression")
    print("="*50)
    
    # Create dataset
    df = create_diabetes_dataset()
    print(f"📊 Dataset created: {df.shape}")
    
    # Perform EDA
    print("\n📈 Performing exploratory data analysis...")
    perform_eda(df)
    
    # Train model
    print("\n🤖 Training logistic regression model...")
    lr_model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba, feature_cols = train_logistic_regression(df)
    
    # Visualize results
    print("\n📊 Creating model visualizations...")
    visualize_model_results(lr_model, X_test_scaled, y_test, y_pred, y_pred_proba, feature_cols)
    
    # Generate insights
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    insights = generate_insights(df, lr_model, feature_cols, accuracy, auc_score)
    
    print("\n" + "="*50)
    print("DIABETES PREDICTION INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")
    
    # Save data and model info
    df.to_csv('diabetes_dataset.csv', index=False)
    
    # Save model coefficients
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': lr_model.coef_[0]})
    coef_df.to_csv('model_coefficients.csv', index=False)
    
    print(f"\n✅ Analysis complete! Check PNG files for visualizations")

if __name__ == "__main__":
    main()