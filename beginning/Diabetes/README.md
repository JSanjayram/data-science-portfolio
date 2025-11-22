# 🩺 Diabetes Prediction using Logistic Regression

## 📋 Problem Statement
Develop a logistic regression model to predict diabetes occurrence based on medical and demographic features, providing insights into risk factors and model performance.

## 🎯 Objectives
- Create realistic diabetes dataset with correlated features
- Perform comprehensive exploratory data analysis
- Train logistic regression classifier for diabetes prediction
- Evaluate model performance using multiple metrics
- Identify key risk factors and feature importance

## 🔍 Approach
1. **Dataset Generation**: Create synthetic diabetes data with realistic correlations
2. **Exploratory Analysis**: Analyze feature distributions and relationships
3. **Model Training**: Implement logistic regression with proper preprocessing
4. **Performance Evaluation**: Use accuracy, AUC, ROC curve, confusion matrix
5. **Feature Analysis**: Identify most important predictive factors

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run diabetes prediction analysis
python diabetes_prediction.py
```

## 📊 Dataset Features

### Medical Indicators
- **Glucose Level**: Blood glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Skin Thickness**: Triceps skin fold thickness (mm)

### Demographic Factors
- **Age**: Age in years
- **Pregnancies**: Number of times pregnant
- **Target**: Diabetes outcome (0=No, 1=Yes)

## 🤖 Machine Learning Pipeline

### Data Preprocessing
- **Feature Scaling**: StandardScaler for logistic regression
- **Train-Test Split**: 70-30 split with stratification
- **Missing Value Handling**: Synthetic data generation ensures completeness

### Model Training
- **Algorithm**: Logistic Regression with L2 regularization
- **Solver**: LBFGS for small datasets
- **Max Iterations**: 1000 for convergence
- **Random State**: 42 for reproducibility

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: True/false positive and negative breakdown
- **Classification Report**: Precision, recall, F1-score per class

## 📈 Visualizations Created

### Exploratory Data Analysis
- Feature distributions by diabetes status
- Correlation heatmap of all features
- Box plots for continuous variables
- Scatter plots for feature relationships

### Model Results
- **Confusion Matrix**: Prediction accuracy breakdown
- **ROC Curve**: True positive vs false positive rates
- **Feature Coefficients**: Logistic regression weights
- **Probability Distribution**: Prediction confidence analysis
- **Feature Importance**: Absolute coefficient values

## 🛠️ Technologies Used
- **Scikit-Learn**: Machine learning algorithms and metrics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations

## 📁 Project Structure
```
Diabetes/
├── diabetes_prediction.py     # Main prediction script
├── requirements.txt          # Dependencies
├── README.md                # Documentation
├── diabetes_dataset.csv     # Generated dataset
├── model_coefficients.csv   # Model weights
├── diabetes_eda.png         # EDA visualizations
└── diabetes_model_results.png # Model performance
```

## 🔍 Key Insights Generated
- Most important risk factors for diabetes prediction
- Model accuracy and reliability metrics
- Feature correlation patterns
- Risk factor thresholds and probabilities
- Population diabetes rate analysis

## 📊 Expected Results
- **Accuracy**: ~75-85% on test set
- **AUC Score**: ~0.80-0.90 (good to excellent)
- **Key Features**: Glucose, BMI, Age typically most important
- **Risk Factors**: High glucose and BMI strongly correlate with diabetes

---
*Built for comprehensive diabetes risk prediction and analysis*