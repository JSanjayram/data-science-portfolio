import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return df

@st.cache_data
def preprocess_data(df):
    # Clean data
    df = df.drop_duplicates()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')
    
    # One-hot encoding
    categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Scale features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    
    return model, scaler, X.columns

def main():
    st.title("🏢 Employee Attrition Prediction System")
    st.markdown("Predict employee attrition and analyze key factors")
    
    # Load data
    df_raw = load_data()
    df = preprocess_data(df_raw)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Prediction", "Analysis"])
    
    if page == "Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", len(df_raw))
        with col2:
            attrition_rate = (df_raw['Attrition'] == 'Yes').mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        with col3:
            st.metric("Features", len(df.columns) - 1)
        
        # Attrition distribution
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        df_raw['Attrition'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Employee Attrition Distribution')
        ax.set_xlabel('Attrition')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Satisfaction correlation
        st.subheader("Satisfaction vs Attrition")
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
        df_temp = df_raw.copy()
        df_temp['Attrition'] = df_temp['Attrition'].map({'Yes': 1, 'No': 0})
        corr_data = df_temp[satisfaction_cols + ['Attrition']].corr()['Attrition'].drop('Attrition')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_data.plot(kind='bar', ax=ax)
        ax.set_title('Correlation between Satisfaction Metrics and Attrition')
        ax.set_ylabel('Correlation')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif page == "Prediction":
        st.header("🔮 Attrition Prediction")
        
        # Train model
        model, scaler, feature_cols = train_model(df)
        
        st.subheader("Enter Employee Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        
        with col2:
            overtime = st.selectbox("Overtime", ["Yes", "No"])
            distance_from_home = st.slider("Distance from Home", 1, 30, 10)
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
        
        if st.button("Predict Attrition"):
            # Create input dataframe (simplified for demo)
            input_data = pd.DataFrame({
                'Age': [age],
                'MonthlyIncome': [monthly_income],
                'YearsAtCompany': [years_at_company],
                'JobSatisfaction': [job_satisfaction],
                'WorkLifeBalance': [work_life_balance],
                'DistanceFromHome': [distance_from_home],
                'EnvironmentSatisfaction': [environment_satisfaction],
                'JobInvolvement': [job_involvement],
                'TotalWorkingYears': [total_working_years]
            })
            
            # Add dummy columns for missing features (simplified approach)
            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns
            input_data = input_data[feature_cols]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Attrition (Probability: {probability[1]:.2f})")
            else:
                st.success(f"✅ Low Risk of Attrition (Probability: {probability[0]:.2f})")
    
    elif page == "Analysis":
        st.header("📈 Feature Importance Analysis")
        
        # Train model and get feature importance
        model, scaler, feature_cols = train_model(df)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        st.subheader("Top 15 Most Important Features")
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = feature_importance.head(15)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance - Random Forest')
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Show feature importance table
        st.subheader("Feature Importance Table")
        st.dataframe(feature_importance.head(20))

if __name__ == "__main__":
    main()