import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Background styling
st.markdown("""
<style>
.stApp {
   // background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   // background: #1101FF;
    background: url('https://t3.ftcdn.net/jpg/05/66/93/78/360_F_566937830_RO0fFVvzUKWIVHCWxUFCs45rMKMuRdYr.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    height: 100vh;
    overflow: hidden;
}
.main .block-container,
[data-testid="stAppViewContainer"] .main .block-container,
.stApp > .main .block-container {
    background: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    padding: 0.5rem 5rem;
    border-radius: 10px;
    margin: 1rem 6rem;
    max-width: 1200px;
    height: calc(100vh - 1rem);
    overflow-y: auto;
    box-sizing: border-box;
    font-size: 0.8rem;
}
.stApp [data-testid="stAppViewContainer"] {
    background: rgba(0, 0, 0, 0.6) !important;
}
.element-container {
    background: transparent !important;
}
h1, h2, h3 {
    color: white !important;
    font-size: 1rem !important;
    margin: 0.2rem 0 !important;
}
h1{
    color: white !important;
    font-size:1.5rem !important;
}
.stMarkdown, .stText, p, div {
    color: white !important;
}
.stSlider > div > div {
    font-size: 0.6rem;
}
.stSelectbox > div > div {
    font-size: 0.9rem;
}
.stMetric {
    font-size: 0.7rem !important;
}
.stMetric > div {
    font-size: 0.9rem !important;
}
.css-1d391kg {
    background: transparent !important;
}
.stSidebar {
    background: transparent !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.stAppHeader {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Load or train models
try:
    model = pickle.load(open('best_student_model.pkl', 'rb'))
    scaler = pickle.load(open('student_scaler.pkl', 'rb'))
    features = pickle.load(open('student_features.pkl', 'rb'))
    
    # Test model compatibility
    test_data = np.zeros((1, len(features)))
    _ = model.predict(test_data)
    model_status = "Online"
except Exception as e:
    model_status = "🔄 Training..."
    st.info("Training model for first time, please wait...")
    
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, 'student.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            model = pickle.load(open('best_student_model.pkl', 'rb'))
            scaler = pickle.load(open('student_scaler.pkl', 'rb'))
            features = pickle.load(open('student_features.pkl', 'rb'))
            model_status = "Online"
            st.success("Model trained successfully!")
            st.rerun()
        else:
            st.error("Training failed. Check if StudentPerformanceFactors.csv exists.")
            st.stop()
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        st.stop()

# Centered title
st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Student Performance Predictor</h1>", unsafe_allow_html=True)

# Sidebar with model information
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model Details")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Target:** Pass/Fail (Threshold: 60)")
st.sidebar.write("**Features:** Study habits and demographics")

st.sidebar.subheader("📈 Performance Metrics")
st.sidebar.write("**Accuracy:** 99.6%")
st.sidebar.write("**ROC-AUC:** 98.4%")
st.sidebar.write("**F1-Score:** 99.8%")
st.sidebar.write("**Precision:** 99.6%")
st.sidebar.write("**Recall:** 99.9%")

st.sidebar.subheader("🔝 Feature Importance")
st.sidebar.write("1. **Study Efficiency** (23.0%)")
st.sidebar.write("2. **Attendance** (19.9%)")
st.sidebar.write("3. **Hours Studied** (19.7%)")
st.sidebar.write("4. **Previous Scores** (15.0%)")
st.sidebar.write("5. **Sleep Hours** (7.2%)")

st.sidebar.subheader("🔄 Implementation Steps")
st.sidebar.write("1. **Data Understanding**")
st.sidebar.write("2. **Data Cleaning**")
st.sidebar.write("3. **Outlier Detection**")
st.sidebar.write("4. **Feature Engineering**")
st.sidebar.write("5. **Encoding**")
st.sidebar.write("6. **Feature Scaling**")
st.sidebar.write("7. **Model Building**")
st.sidebar.write("8. **Model Evaluation**")
st.sidebar.write("9. **Hyperparameter Tuning**")
st.sidebar.write("10. **Model Interpretation**")



# Input form
st.header("Enter Student Details")

# Create columns for input and results
input_col1, input_col2, result_col = st.columns([1, 1, 1])

with input_col1:
    hours_studied = st.slider("Hours Studied per Week", 0, 50, 20)
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    sleep_hours = st.slider("Sleep Hours per Day", 4, 12, 8)
    previous_scores = st.slider("Previous Scores", 0, 100, 75)
    tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 20, 5)

with input_col2:
   # tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 20, 5)
    physical_activity = st.slider("Physical Activity Hours per Week", 0, 20, 5)
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    parental_education = st.selectbox("Parental Education Level", ["Low", "Medium", "High"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    predict_clicked = st.button("🔮 Predict Performance", type="primary")

with result_col:
    st.markdown("### 🎯 Results")
    #st.markdown("<br><br>", unsafe_allow_html=True)
    
if predict_clicked:
    # Input validation
    if hours_studied == 0 and attendance == 0 and previous_scores == 0:
        st.error("❌ Invalid Input: Please enter realistic values for study hours, attendance, and previous scores.")
        st.stop()
    
    # Calculate derived features
    study_efficiency = previous_scores / (hours_studied + 1) if hours_studied > 0 else 0
    gender_encoded = 1 if gender == "Male" else 0
    
    # Create base input data
    input_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,
        'Study_Efficiency': study_efficiency,
        'Gender_Encoded': gender_encoded
    }
    
    # Add categorical features as dummy variables
    for feature in features:
        if feature not in input_data:
            # Set default values for categorical features
            if 'Parental_Involvement' in feature:
                input_data[feature] = 1 if parental_involvement.lower() in feature.lower() else 0
            elif 'Parental_Education_Level' in feature:
                input_data[feature] = 1 if parental_education.lower() in feature.lower() else 0
            else:
                input_data[feature] = 0
    
    # Create DataFrame with exact feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Apply proper scaling using the saved scaler
    numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                     'Study_Efficiency', 'Tutoring_Sessions', 'Physical_Activity', 'Gender_Encoded']
    
    # Create a full feature array for scaling (8 features to match scaler)
    scale_data = np.array([[hours_studied, attendance, sleep_hours, previous_scores, 
                           study_efficiency, tutoring_sessions, physical_activity, gender_encoded]])
    
    # Scale the data
    scaled_data = scaler.transform(scale_data)
    
    # Update the input dataframe with scaled values
    for i, col in enumerate(numerical_cols):
        if col in input_df.columns:
            input_df[col] = scaled_data[0][i]
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results in the result column
        with result_col:
            if prediction == 1:
                st.markdown("<div style='font-size: 0.8rem; color: green;'>🎉 PASS</div>", unsafe_allow_html=True)
                st.metric("Success", f"{probability[1]:.1%}")
            else:
                st.markdown("<div style='font-size: 0.8rem; color: red;'>❌ FAIL</div>", unsafe_allow_html=True)
                st.metric("Failure", f"{probability[0]:.1%}")
            
            # Risk factors
            risk_factors = []
            if hours_studied < 15:
                risk_factors.append("Low study hours")
            if attendance < 80:
                risk_factors.append("Poor attendance")
            if sleep_hours < 6:
                risk_factors.append("Insufficient sleep")
            if previous_scores < 60:
                risk_factors.append("Low previous scores")
            
            if risk_factors:
                st.markdown("<div style='font-size: 0.7rem; color: orange;'>⚠️ Risk Factors:</div>", unsafe_allow_html=True)
                for factor in risk_factors:
                    st.markdown(f"<div style='font-size: 0.6rem;'>• {factor}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 0.7rem; color: green;'>✅ Good habits!</div>", unsafe_allow_html=True)
        
            # Grid layout for analysis
            st.markdown("<div style='font-size: 0.8rem; font-weight: bold;'>📊 Analysis</div>", unsafe_allow_html=True)
            
            # Create 2x3 grid
            grid_col1, grid_col2 = st.columns(2)
            
            with grid_col1:
                st.metric("Study Hours", hours_studied, delta="/week")
                st.metric("Study Efficiency", f"{study_efficiency:.2f}")
                st.markdown(f"<div style='font-size: 0.6rem;'><b>Parental Involvement:</b> {parental_involvement}</div>", unsafe_allow_html=True)
            
            with grid_col2:
                st.metric("Attendance", f"{attendance}%")
                st.metric("Sleep Hours", sleep_hours, delta="/day")
                st.markdown(f"<div style='font-size: 0.6rem;'><b>Parental Education:</b> {parental_education}</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
