import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page config with custom styling
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢", layout="wide")

# Custom CSS with SVG animated background
st.markdown("""
<style>
    .stApp {
        background-image: url("data:image/svg+xml,%3Csvg width='100%25' height='100%25' id='svg' viewBox='0 0 1440 590' xmlns='http://www.w3.org/2000/svg' class='transition duration-300 ease-in-out delay-150'%3E%3Cstyle%3E .path-0%7B animation:pathAnim-0 4s; animation-timing-function: linear; animation-iteration-count: infinite; %7D @keyframes pathAnim-0%7B 0%25%7B d: path('M 0,600 L 0,150 C 75.59808612440187,152.46889952153109 151.19617224880375,154.93779904306217 256,174 C 360.80382775119625,193.06220095693783 494.8133971291867,228.71770334928232 603,254 C 711.1866028708133,279.2822966507177 793.5502392344497,294.19138755980856 890,324 C 986.4497607655503,353.80861244019144 1096.9856459330144,398.51674641148327 1191,422 C 1285.0143540669856,445.48325358851673 1362.5071770334928,447.74162679425837 1440,450 L 1440,600 L 0,600 Z'); %7D 25%25%7B d: path('M 0,600 L 0,150 C 106.26794258373207,169.54066985645932 212.53588516746413,189.08133971291866 301,191 C 389.46411483253587,192.91866028708134 460.12440191387554,177.2153110047847 551,201 C 641.8755980861245,224.7846889952153 752.9665071770336,288.0574162679426 862,321 C 971.0334928229664,353.9425837320574 1078.0095693779904,356.555023923445 1174,373 C 1269.9904306220096,389.444976076555 1354.9952153110048,419.7224880382775 1440,450 L 1440,600 L 0,600 Z'); %7D 50%25%7B d: path('M 0,600 L 0,150 C 77.90430622009566,152.8421052631579 155.80861244019133,155.68421052631578 241,155 C 326.1913875598087,154.31578947368422 418.66985645933016,150.10526315789477 537,190 C 655.3301435406698,229.89473684210523 799.5119617224882,313.89473684210526 914,338 C 1028.4880382775118,362.10526315789474 1113.2822966507176,326.3157894736842 1196,335 C 1278.7177033492824,343.6842105263158 1359.3588516746413,396.8421052631579 1440,450 L 1440,600 L 0,600 Z'); %7D 75%25%7B d: path('M 0,600 L 0,150 C 63.626794258373195,168.79425837320574 127.25358851674639,187.58851674641147 240,182 C 352.7464114832536,176.41148325358853 514.6124401913876,146.44019138755982 614,158 C 713.3875598086124,169.55980861244018 750.2966507177035,222.65071770334924 833,284 C 915.7033492822965,345.34928229665076 1044.200956937799,414.9569377990431 1153,444 C 1261.799043062201,473.0430622009569 1350.8995215311006,461.52153110047846 1440,450 L 1440,600 L 0,600 Z'); %7D 100%25%7B d: path('M 0,600 L 0,150 C 75.59808612440187,152.46889952153109 151.19617224880375,154.93779904306217 256,174 C 360.80382775119625,193.06220095693783 494.8133971291867,228.71770334928232 603,254 C 711.1866028708133,279.2822966507177 793.5502392344497,294.19138755980856 890,324 C 986.4497607655503,353.80861244019144 1096.9856459330144,398.51674641148327 1191,422 C 1285.0143540669856,445.48325358851673 1362.5071770334928,447.74162679425837 1440,450 L 1440,600 L 0,600 Z'); %7D %7D .path-1%7B animation:pathAnim-1 4s; animation-timing-function: linear; animation-iteration-count: infinite; %7D @keyframes pathAnim-1%7B 0%25%7B d: path('M 0,600 L 0,350 C 109.3397129186603,317.444976076555 218.6794258373206,284.8899521531101 317,314 C 415.3205741626794,343.1100478468899 502.62200956937795,433.88516746411483 586,457 C 669.377990430622,480.11483253588517 748.8325358851677,435.5693779904306 831,437 C 913.1674641148323,438.4306220095694 998.0478468899521,485.8373205741626 1100,529 C 1201.952153110048,572.1626794258374 1320.976076555024,611.0813397129186 1440,650 L 1440,600 L 0,600 Z'); %7D 25%25%7B d: path('M 0,600 L 0,350 C 103.15789473684211,363.5980861244019 206.31578947368422,377.19617224880386 305,378 C 403.6842105263158,378.80382775119614 497.8947368421052,366.8133971291866 583,369 C 668.1052631578948,371.1866028708134 744.1052631578948,387.55023923444975 851,437 C 957.8947368421052,486.44976076555025 1095.6842105263158,568.9856459330143 1199,610 C 1302.3157894736842,651.0143540669857 1371.157894736842,650.5071770334928 1440,650 L 1440,600 L 0,600 Z'); %7D 50%25%7B d: path('M 0,600 L 0,350 C 77.60765550239233,355.82775119617224 155.21531100478467,361.65550239234454 246,369 C 336.78468899521533,376.34449760765546 440.7464114832536,385.2057416267942 546,418 C 651.2535885167464,450.7942583732058 757.7990430622008,507.52153110047857 856,523 C 954.2009569377992,538.4784688995214 1044.0574162679427,512.7081339712919 1140,527 C 1235.9425837320573,541.2918660287081 1337.9712918660287,595.6459330143541 1440,650 L 1440,600 L 0,600 Z'); %7D 75%25%7B d: path('M 0,600 L 0,350 C 79.19617224880383,364.9569377990431 158.39234449760767,379.9138755980861 267,393 C 375.60765550239233,406.0861244019139 513.6267942583731,417.30143540669854 600,451 C 686.3732057416269,484.69856459330146 721.1004784688996,540.8803827751196 819,543 C 916.8995215311004,545.1196172248804 1077.9712918660289,493.177033492823 1192,502 C 1306.0287081339711,510.822966507177 1373.0143540669856,580.4114832535885 1440,650 L 1440,600 L 0,600 Z'); %7D 100%25%7B d: path('M 0,600 L 0,350 C 109.3397129186603,317.444976076555 218.6794258373206,284.8899521531101 317,314 C 415.3205741626794,343.1100478468899 502.62200956937795,433.88516746411483 586,457 C 669.377990430622,480.11483253588517 748.8325358851677,435.5693779904306 831,437 C 913.1674641148323,438.4306220095694 998.0478468899521,485.8373205741626 1100,529 C 1201.952153110048,572.1626794258374 1320.976076555024,611.0813397129186 1440,650 L 1440,600 L 0,600 Z'); %7D %7D %3C/style%3E%3Cdefs%3E%3ClinearGradient id='gradient' x1='0%25' y1='50%25' x2='100%25' y2='50%25'%3E%3Cstop offset='5%25' stop-color='%23F78DA7'%3E%3C/stop%3E%3Cstop offset='95%25' stop-color='%238ED1FC'%3E%3C/stop%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M 0,600 L 0,150 C 75.59808612440187,152.46889952153109 151.19617224880375,154.93779904306217 256,174 C 360.80382775119625,193.06220095693783 494.8133971291867,228.71770334928232 603,254 C 711.1866028708133,279.2822966507177 793.5502392344497,294.19138755980856 890,324 C 986.4497607655503,353.80861244019144 1096.9856459330144,398.51674641148327 1191,422 C 1285.0143540669856,445.48325358851673 1362.5071770334928,447.74162679425837 1440,450 L 1440,600 L 0,600 Z' stroke='none' stroke-width='0' fill='url(%23gradient)' fill-opacity='0.53' class='transition-all duration-300 ease-in-out delay-150 path-0'%3E%3C/path%3E%3Cpath d='M 0,600 L 0,350 C 109.3397129186603,317.444976076555 218.6794258373206,284.8899521531101 317,314 C 415.3205741626794,343.1100478468899 502.62200956937795,433.88516746411483 586,457 C 669.377990430622,480.11483253588517 748.8325358851677,435.5693779904306 831,437 C 913.1674641148323,438.4306220095694 998.0478468899521,485.8373205741626 1100,529 C 1201.952153110048,572.1626794258374 1320.976076555024,611.0813397129186 1440,650 L 1440,600 L 0,600 Z' stroke='none' stroke-width='0' fill='url(%23gradient)' fill-opacity='1' class='transition-all duration-300 ease-in-out delay-150 path-1'%3E%3C/path%3E%3C/svg%3E");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .main {
        background: transparent;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train_model():
    """Load data and train model"""
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df = df.drop_duplicates()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')
    
    categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    
    X = df.drop('Attrition', axis=1).astype(float)
    y = df['Attrition'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist(), df

def main():
    # Header with gradient background
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🏢 Employee Attrition Prediction System</h1>
        <p style="color: white; margin: 0; font-size: 1.2rem;">AI-Powered HR Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler, feature_names, df = load_and_train_model()
    
    # Sidebar with custom styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📊 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose a page", ["🔮 Prediction", "📊 Overview", "🔍 Insights"])
    
    if page == "🔮 Prediction":
        st.markdown("### 🎯 Predict Employee Attrition Risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            age = st.slider("Age", 18, 65, 30)
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            distance_from_home = st.slider("Distance from Home (miles)", 1, 30, 10)
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
        
        with col2:
            st.markdown("#### 💼 Job & Satisfaction")
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            job_level = st.slider("Job Level", 1, 5, 2)
        
        if st.button("🎯 Predict Attrition Risk", type="primary"):
            # Create input array
            input_data = np.zeros(len(feature_names))
            
            feature_mapping = {
                'Age': age, 'MonthlyIncome': monthly_income, 'DistanceFromHome': distance_from_home,
                'TotalWorkingYears': total_working_years, 'YearsAtCompany': years_at_company,
                'JobSatisfaction': job_satisfaction, 'EnvironmentSatisfaction': environment_satisfaction,
                'WorkLifeBalance': work_life_balance, 'JobLevel': job_level,
                'DailyRate': 800, 'HourlyRate': 65, 'MonthlyRate': 14000, 'Education': 3,
                'JobInvolvement': 3, 'NumCompaniesWorked': 2, 'PercentSalaryHike': 15,
                'PerformanceRating': 3, 'RelationshipSatisfaction': 3, 'StockOptionLevel': 1,
                'TrainingTimesLastYear': 3, 'YearsInCurrentRole': min(years_at_company, 4),
                'YearsSinceLastPromotion': max(0, years_at_company - 2),
                'YearsWithCurrManager': min(years_at_company, 3)
            }
            
            for i, feature in enumerate(feature_names):
                if feature in feature_mapping:
                    input_data[i] = feature_mapping[feature]
                elif feature == 'OverTime_Yes':
                    input_data[i] = 1 if overtime == 'Yes' else 0
                else:
                    input_data[i] = 0
            
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-high">
                        <h3>⚠️ HIGH RISK of Attrition</h3>
                        <h2>{probability[1]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-low">
                        <h3>✅ LOW RISK of Attrition</h3>
                        <h2>{probability[0]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🚨 Risk Factors")
                risk_factors = []
                if monthly_income < 5000: risk_factors.append("💰 Low monthly income")
                if age < 30: risk_factors.append("👶 Young age")
                if overtime == "Yes": risk_factors.append("⏰ Overtime work")
                if distance_from_home > 15: risk_factors.append("🚗 Long commute")
                if job_satisfaction <= 2: risk_factors.append("😞 Low job satisfaction")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("✨ No major risk factors identified")
    
    elif page == "📊 Overview":
        st.markdown("### 📈 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Employees</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            attrition_rate = df['Attrition'].mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>📉 Attrition Rate</h3>
                <h2>{attrition_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢 Features</h3>
                <h2>{len(feature_names)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Attrition Distribution")
        attrition_counts = df['Attrition'].value_counts()
        st.bar_chart(attrition_counts)
    
    elif page == "🔍 Insights":
        st.markdown("### 🔍 Model Insights")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.markdown("#### 🏆 Top 15 Most Important Features")
        top_features = feature_importance.head(15)
        st.bar_chart(top_features.set_index('feature')['importance'])
        
        st.markdown("#### 📋 Feature Importance Table")
        st.dataframe(feature_importance.head(20), use_container_width=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h4>🤖 Model Information</h4>
            <p><strong>Model Type:</strong> Random Forest with Balanced Class Weights</p>
            <p><strong>Number of Trees:</strong> 100</p>
            <p><strong>Features Used:</strong> """ + str(len(feature_names)) + """</p>
            <p><strong>Training Strategy:</strong> Handles class imbalance with balanced weights</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()