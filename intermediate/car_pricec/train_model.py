import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load and clean data
df = pd.read_csv('car_dataset (1).csv')
df.dropna(inplace=True)

# Feature engineering
current_year = datetime.now().year
df['car_age'] = current_year - df['model_year']
df['Price_per_KM'] = df['price'] / (df['kilometers_run'] + 1)

# Age categories
df['Age_Category'] = pd.cut(df['car_age'], bins=[-1, 3, 8, 100], labels=['New', 'Mid', 'Old'])

# Label encoding for high cardinality features
le_brand = LabelEncoder()
le_model = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['brand'])
df['car_model_encoded'] = le_model.fit_transform(df['car_model'])

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['transmission', 'body_type', 'fuel_type', 'Age_Category'])

# Select features
feature_cols = ['model_year', 'engine_capacity', 'kilometers_run', 'Price_per_KM', 
                'brand_encoded', 'car_model_encoded'] + [col for col in df_encoded.columns if col.startswith(('transmission_', 'body_type_', 'fuel_type_', 'Age_Category_'))]

X = df_encoded[feature_cols].fillna(0)
y = df_encoded['price'] / 1000000  # Scale target

# Train model with explicit parameters for compatibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train)

# Save files
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le_brand, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained and saved successfully!")
print(f"Model score: {model.score(X_test, y_test):.3f}")