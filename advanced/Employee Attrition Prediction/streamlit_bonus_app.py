import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢", layout="wide")

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # Data cleaning
    df = df.drop_duplicates()
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, errors='ignore')
    
    # Feature engineering
    df['YearsSinceLastPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
    df['OverTime_Hours'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['TotalSatisfaction'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                              df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
    df['IncomePerYear'] = df['MonthlyIncome'] * 12
    df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # Encoding
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

@st.cache_resource
def train_model(df):
    """Train the model"""
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Scale features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

def main():
    st.title("🏢 Employee Attrition Prediction System")
    st.markdown("**Bonus Task: Streamlit App for Employee Attrition Prediction**")
    
    # Load data and train model
    df = load_and_prepare_data()
    model, scaler, feature_columns = train_model(df)
    
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Dataset Overview", "Model Insights"])
    
    if page == "Prediction":
        st.header("🔮 Predict Employee Attrition")
        st.markdown("Enter employee details to predict attrition risk:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Info")
            age = st.slider("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.slider("Distance from Home (miles)", 1, 30, 10)
        
        with col2:
            st.subheader("Job Details")
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", 
                                               "Manufacturing Director", "Healthcare Representative", "Manager", 
                                               "Sales Representative", "Research Director", "Human Resources"])
            job_level = st.slider("Job Level", 1, 5, 2)
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            overtime = st.selectbox("Overtime", ["Yes", "No"])
        
        with col3:
            st.subheader("Experience & Satisfaction")
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
            environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
            work_life_balance = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
        
        if st.button("🎯 Predict Attrition Risk", type="primary"):
            # Create input dataframe with engineered features
            input_data = {
                'Age': age,
                'DailyRate': 800,  # Default value
                'DistanceFromHome': distance_from_home,
                'Education': 3,  # Default
                'EnvironmentSatisfaction': environment_satisfaction,
                'HourlyRate': 65,  # Default
                'JobInvolvement': 3,  # Default
                'JobLevel': job_level,
                'JobSatisfaction': job_satisfaction,
                'MonthlyIncome': monthly_income,
                'MonthlyRate': 14000,  # Default
                'NumCompaniesWorked': 2,  # Default
                'PercentSalaryHike': 15,  # Default
                'PerformanceRating': 3,  # Default
                'RelationshipSatisfaction': 3,  # Default
                'StockOptionLevel': 1,  # Default
                'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': 3,  # Default
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': min(years_at_company, 4),
                'YearsSinceLastPromotion': max(0, years_at_company - 2),
                'YearsWithCurrManager': min(years_at_company, 3),
                
                # Engineered features
                'OverTime_Hours': 1 if overtime == 'Yes' else 0,
                'TotalSatisfaction': (job_satisfaction + environment_satisfaction + 3 + work_life_balance) / 4,
                'IncomePerYear': monthly_income * 12,
                'ExperienceRatio': years_at_company / (total_working_years + 1)
            }
            
            # Create dummy variables for categorical features
            for col in feature_columns:
                if col not in input_data:
                    if any(x in col for x in ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']):
                        input_data[col] = 0
                    else:
                        input_data[col] = 0
            
            # Set specific categorical encodings
            if f'Gender_{gender}' in feature_columns:
                input_data[f'Gender_{gender}'] = 1
            if f'Department_{department}' in feature_columns:
                input_data[f'Department_{department}'] = 1
            if f'OverTime_{overtime}' in feature_columns:
                input_data[f'OverTime_{overtime}'] = 1
            if f'MaritalStatus_{marital_status}' in feature_columns:
                input_data[f'MaritalStatus_{marital_status}'] = 1
            if f'JobRole_{job_role}' in feature_columns:
                input_data[f'JobRole_{job_role}'] = 1
            
            # Create DataFrame and ensure correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)
            
            # Scale numerical features
            numerical_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK** of Attrition")
                    st.error(f"Probability: {probability[1]:.1%}")
                else:
                    st.success(f"✅ **LOW RISK** of Attrition")
                    st.success(f"Probability: {probability[0]:.1%}")
            
            with col2:
                # Risk factors
                st.subheader("Risk Factors Analysis")
                risk_factors = []
                if monthly_income < 5000:
                    risk_factors.append("💰 Low monthly income")
                if age < 30:
                    risk_factors.append("👶 Young age")
                if overtime == "Yes":
                    risk_factors.append("⏰ Overtime work")
                if distance_from_home > 15:
                    risk_factors.append("🚗 Long commute")
                if job_satisfaction <= 2:
                    risk_factors.append("😞 Low job satisfaction")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.info("✨ No major risk factors identified")
    
    elif page == "Dataset Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            attrition_rate = df['Attrition'].mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        with col3:
            st.metric("Features", len(df.columns) - 1)
        
        # Attrition distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        df['Attrition'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Attrition Distribution')
        ax.set_xlabel('Attrition (0=No, 1=Yes)')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['No Attrition', 'Attrition'], rotation=0)
        st.pyplot(fig)
    
    elif page == "Model Insights":
        st.header("🔍 Model Insights")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.subheader("Top 15 Most Important Features")
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(15)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Feature Importance - Random Forest')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            st.subheader("Feature Importance Table")
            st.dataframe(feature_importance.head(20))

if __name__ == "__main__":
    main()