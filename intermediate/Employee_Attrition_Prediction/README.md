# Employee Attrition Prediction (ANN) 🧠

## Overview
This project predicts whether an employee will stay or leave a company based on HR-related data using an Artificial Neural Network (ANN).

## 🎯 Objective
Predict employee attrition (Yes/No) to help HR take preventive actions and improve retention.

## 🧰 Tools / Libraries
- Python
- TensorFlow/Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib/Seaborn
- Streamlit

## 🧾 Dataset Features
- Age
- Department
- JobRole
- MonthlyIncome
- Overtime
- YearsAtCompany
- JobSatisfaction
- WorkLifeBalance

## 🧮 Model Architecture
```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## 💡 Expected Output
Predicts attrition probability, e.g.:
- **Predicted Attrition Probability: 0.78** → Employee likely to leave
- **Predicted Attrition Probability: 0.23** → Employee likely to stay

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Generate Sample Data**
```bash
python data_generator.py
```

3. **Run Application**
```bash
streamlit run app.py
```

## 📊 Features
- **Real-time Prediction**: Input employee data and get instant attrition probability
- **Risk Analysis**: Identifies key risk factors for employee departure
- **Data Dashboard**: Visualize attrition patterns and trends
- **Model Information**: View model architecture and performance metrics

## 🚀 Possible Extensions
- Add HR dashboards with advanced analytics
- Implement SHAP/LIME explanations for model interpretability
- Integrate into existing HR software systems
- Add employee retention recommendations
- Implement real-time monitoring and alerts