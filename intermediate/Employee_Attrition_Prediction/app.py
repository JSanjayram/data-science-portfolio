import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attrition_model import AttritionPredictor
from data_generator import generate_sample_data

# Page config
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main { padding: 2rem; }
.stMetric { background: #f0f2f6; padding: 1rem; border-radius: 10px; }
.prediction-box { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 2rem; border-radius: 15px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        return pd.read_csv('employee_data.csv')
    except:
        return generate_sample_data(1000)

@st.cache_resource
def train_model():
    df = load_data()
    predictor = AttritionPredictor()
    X, y = predictor.preprocess_data(df)
    accuracy, _, _ = predictor.train(X, y)
    return predictor, accuracy

def main():
    st.title("🧠 Employee Attrition Prediction System")
    st.markdown("Predict employee attrition using Artificial Neural Networks")
    
    # Load model
    predictor, accuracy = train_model()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Info"])
    
    if page == "Prediction":
        st.header("🎯 Employee Attrition Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Employee Information")
            
            age = st.slider("Age", 18, 65, 30)
            monthly_income = st.slider("Monthly Income ($)", 2000, 15000, 5000)
            years_at_company = st.slider("Years at Company", 0.0, 30.0, 5.0)
            
            job_role = st.selectbox("Job Role", 
                ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                 'Manufacturing Director', 'Healthcare Representative', 'Manager'])
            
            department = st.selectbox("Department", 
                ['Sales', 'Research & Development', 'Human Resources'])
            
            overtime = st.selectbox("Overtime", ['Yes', 'No'])
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.button("🔮 Predict Attrition", type="primary"):
                # Prepare input data
                employee_data = {
                    'Age': age,
                    'MonthlyIncome': monthly_income,
                    'YearsAtCompany': years_at_company,
                    'JobRole': job_role,
                    'Department': department,
                    'OverTime': overtime,
                    'JobSatisfaction': job_satisfaction,
                    'WorkLifeBalance': work_life_balance
                }
                
                # Make prediction
                probability = predictor.predict_attrition(employee_data)
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Attrition Probability: {probability:.2%}</h2>
                    <h3>{'🚨 High Risk - Employee Likely to Leave' if probability > 0.5 else '✅ Low Risk - Employee Likely to Stay'}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors
                st.subheader("Risk Analysis")
                risk_factors = []
                
                if age < 25: risk_factors.append("Young age (higher turnover)")
                if monthly_income < 3500: risk_factors.append("Below average salary")
                if overtime == 'Yes': risk_factors.append("Overtime work")
                if job_satisfaction <= 2: risk_factors.append("Low job satisfaction")
                if work_life_balance <= 2: risk_factors.append("Poor work-life balance")
                if years_at_company < 2: risk_factors.append("New employee")
                
                if risk_factors:
                    st.warning("Risk Factors Identified:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No major risk factors identified!")
    
    elif page == "Data Analysis":
        st.header("📊 Data Analysis Dashboard")
        
        df = load_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            attrition_rate = (df['Attrition'] == 'Yes').mean()
            st.metric("Attrition Rate", f"{attrition_rate:.1%}")
        with col3:
            avg_age = df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        with col4:
            avg_income = df['MonthlyIncome'].mean()
            st.metric("Avg Income", f"${avg_income:,.0f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Attrition by Department")
            fig, ax = plt.subplots(figsize=(8, 6))
            attrition_dept = df.groupby(['Department', 'Attrition']).size().unstack()
            attrition_dept.plot(kind='bar', ax=ax, color=['lightgreen', 'salmon'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Age Distribution by Attrition")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x='Attrition', y='Age', ax=ax)
            st.pyplot(fig)
    
    else:  # Model Info
        st.header("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.code("""
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
            """)
            
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{accuracy:.1%}")
        
        with col2:
            st.subheader("Features Used")
            features = ['Age', 'Monthly Income', 'Years at Company', 'Job Role', 
                       'Department', 'Overtime', 'Job Satisfaction', 'Work Life Balance']
            
            for feature in features:
                st.write(f"• {feature}")
            
            st.subheader("Model Details")
            st.write("• **Algorithm**: Artificial Neural Network")
            st.write("• **Framework**: TensorFlow/Keras")
            st.write("• **Activation**: ReLU (hidden), Sigmoid (output)")
            st.write("• **Loss Function**: Binary Crossentropy")
            st.write("• **Optimizer**: Adam")

if __name__ == "__main__":
    main()