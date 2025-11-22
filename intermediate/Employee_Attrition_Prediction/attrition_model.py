import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class AttritionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        # Create target variable
        df['Attrition_Binary'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Select key features
        features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'OverTime', 
                   'JobRole', 'Department', 'WorkLifeBalance', 'JobSatisfaction']
        
        X = df[features].copy()
        y = df['Attrition_Binary']
        
        # Encode categorical variables
        categorical_cols = ['OverTime', 'JobRole', 'Department']
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        return X, y
    
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(input_shape,)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        if self.model is None:
            self.build_model(X_train.shape[1])
            
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
        
        # Evaluate
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, X_test, y_test
    
    def predict_attrition(self, employee_data):
        if self.model is None:
            return None
            
        # Encode categorical data
        for col, encoder in self.label_encoders.items():
            if col in employee_data:
                employee_data[col] = encoder.transform([str(employee_data[col])])[0]
        
        # Scale features
        features_array = np.array(list(employee_data.values())).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        probability = self.model.predict(features_scaled)[0][0]
        return probability