# Student Performance Prediction

## Overview
Machine learning system that predicts student pass/fail outcomes with 99.6% accuracy using Random Forest classifier.

## Features
- **High Accuracy**: 99.6% prediction accuracy
- **Web Interface**: Interactive Streamlit dashboard
- **Real-time Predictions**: Instant pass/fail probability
- **Risk Analysis**: Identifies performance risk factors

## Quick Start
```bash
pip install -r requirements.txt
streamlit run student_app.py
```

## Model Performance
- **Accuracy**: 99.6%
- **ROC-AUC**: 98.4%
- **F1-Score**: 99.8%

## Key Features
1. Study Efficiency (23.0%)
2. Attendance (19.9%)
3. Hours Studied (19.7%)
4. Previous Scores (15.0%)
5. Sleep Hours (7.2%)

## Files
- `student.py` - Model training
- `student_app.py` - Web application
- `StudentPerformanceFactors.csv` - Dataset
- `requirements.txt` - Dependencies

## 🔗 Live Demo
[Student Performance Predictor](https://studentmlmodel.streamlit.app/)

## Usage
Enter student details in the web interface to get pass/fail prediction with probability scores and risk assessment.