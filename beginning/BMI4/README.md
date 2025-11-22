# ⚖️ BMI Calculator Web App (Streamlit)

## 📋 Problem Statement
Create an interactive web application to calculate Body Mass Index (BMI) and provide health insights based on user input for weight and height measurements.

## 🎯 Objectives
- Build user-friendly BMI calculator interface
- Support both metric and imperial unit systems
- Provide visual BMI category representation
- Generate personalized health recommendations
- Track BMI history and trends over time
- Display comprehensive health information

## 🔍 Approach
1. **Web Interface**: Interactive Streamlit application
2. **Dual Units**: Support metric (kg/m) and imperial (lbs/ft) systems
3. **Visual Charts**: BMI category visualization with current position
4. **Health Insights**: Category-based recommendations
5. **History Tracking**: Session-based BMI trend monitoring

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run bmi_app.py
```

## 📊 Features

### Core Functionality
- **BMI Calculation**: Accurate BMI computation using standard formula
- **Unit Conversion**: Automatic metric/imperial conversion
- **Category Classification**: Underweight, Normal, Overweight, Obese
- **Visual Indicators**: Color-coded status with emojis

### Interactive Elements
- **Sidebar Inputs**: Clean input interface for measurements
- **Real-time Calculation**: Instant BMI computation on button click
- **Responsive Design**: Wide layout with organized columns
- **Visual Charts**: Horizontal BMI range chart with position marker

### Health Insights
- **Category Status**: Clear BMI category identification
- **Health Recommendations**: Personalized advice based on BMI
- **Status Alerts**: Color-coded health status indicators
- **Educational Content**: BMI information and guidelines

### Advanced Features
- **BMI History**: Session-based tracking of calculations
- **Trend Analysis**: Line chart showing BMI progression
- **Data Export**: Tabular history display
- **Information Panel**: Expandable BMI education section

## 📈 BMI Categories
- **🔵 Underweight**: BMI < 18.5
- **🟢 Normal**: BMI 18.5-24.9
- **🟡 Overweight**: BMI 25-29.9
- **🔴 Obese**: BMI ≥ 30

## 🛠️ Technologies Used
- **Streamlit**: Web application framework
- **Matplotlib**: BMI chart visualization
- **Pandas**: Data handling and history tracking
- **NumPy**: Numerical computations

## 📁 Project Structure
```
BMI4/
├── bmi_app.py          # Main Streamlit application
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## 🎨 User Interface
- **Sidebar**: Input controls and unit selection
- **Main Panel**: Results display and visualizations
- **Metrics Row**: BMI value, category, and health status
- **Chart Section**: Visual BMI range representation
- **Recommendations**: Health advice based on category
- **History Panel**: BMI tracking and trends

## 🔗 Usage Instructions
1. Select unit system (Metric/Imperial)
2. Enter weight and height measurements
3. Click "Calculate BMI" button
4. View results, chart, and recommendations
5. Track history for multiple calculations

## 📊 Health Recommendations
- **Underweight**: Increase caloric intake, strength training
- **Normal**: Maintain lifestyle, regular exercise
- **Overweight**: Moderate calorie reduction, increase activity
- **Obese**: Consult healthcare provider, lifestyle changes

---
*Built for comprehensive BMI calculation and health monitoring*