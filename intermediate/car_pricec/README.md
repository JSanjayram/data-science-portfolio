# Car Price Prediction System

## 🚗 Overview
A machine learning-powered web application that predicts car prices based on various features using Random Forest Regressor with 95.86% accuracy.

## 🎯 Features
- **Interactive Web Interface** - Built with Streamlit
- **Real-time Predictions** - Instant price estimates
- **Advanced ML Model** - Random Forest with 42 engineered features
- **Smart Filtering** - Dynamic options based on brand/model selection
- **Professional UI** - Car-themed background with transparent design

## 📊 Model Performance
- **Algorithm**: Random Forest Regressor
- **Accuracy**: 95.86% (R² Score)
- **Training Data**: 1,132 records (after cleaning)
- **Features**: 42 engineered features
- **Top Feature**: Price_per_KM (73.8% importance)

## 🔧 Key Features Analyzed
- Engine Capacity
- Kilometers Run
- Model Year
- Brand & Car Model
- Transmission Type
- Body & Fuel Type
- Car Age Category

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

## 📁 Project Structure
```
car_pricec/
├── app.py                 # Main Streamlit application
├── train_model_v2.py      # Model training script
├── car_dataset (1).csv    # Training dataset
├── model.pkl              # Trained model
├── scaler.pkl             # Feature scaler
├── label_encoder.pkl      # Categorical encoder
├── feature_names.pkl      # Feature names list
└── requirements.txt       # Dependencies
```

## 🛠️ Technical Details

### Data Processing Pipeline
1. **Data Loading** - CSV import and validation
2. **Data Cleaning** - Remove nulls and duplicates
3. **Outlier Removal** - IQR method for price outliers
4. **Feature Engineering** - Price_per_KM, Age_Category creation
5. **Encoding** - Label encoding for brands/models, one-hot for categories
6. **Model Training** - Random Forest with optimized hyperparameters

### Model Architecture
- **Base Algorithm**: Random Forest Regressor
- **Hyperparameters**: 
  - n_estimators=100
  - max_depth=15
  - min_samples_split=5
  - min_samples_leaf=2
- **Cross-validation**: 80/20 train-test split

## 📈 Performance Metrics
- **Training R² Score**: 98.50%
- **Testing R² Score**: 95.86%
- **Feature Importance**: Price_per_KM dominates at 73.8%

## 🔗 Live Demo
[Car Price Predictor](https://carmlmodel.streamlit.app/)

## 📋 Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn==1.5.1

## 📄 License
MIT License - see LICENSE file for details