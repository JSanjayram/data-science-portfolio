import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Loading and processing data...")

# Load and clean data
df = pd.read_csv('car_dataset (1).csv')
print(f"Original dataset shape: {df.shape}")

# Clean data
df = df.dropna()
print(f"After removing nulls: {df.shape}")

# Remove outliers using IQR method for price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print(f"After removing outliers: {df.shape}")

# Feature engineering
current_year = datetime.now().year
df['car_age'] = current_year - df['model_year']
df['Price_per_KM'] = df['price'] / (df['kilometers_run'] + 1)

# Age categories
df['Age_Category'] = pd.cut(df['car_age'], bins=[-1, 3, 8, 100], labels=['New', 'Mid', 'Old'])

print("Encoding categorical variables...")

# Label encoding for high cardinality features
le_brand = LabelEncoder()
le_model = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['brand'])
df['car_model_encoded'] = le_model.fit_transform(df['car_model'])

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['transmission', 'body_type', 'fuel_type', 'Age_Category'])

# Select features for training
feature_cols = ['model_year', 'engine_capacity', 'kilometers_run', 'Price_per_KM', 
                'brand_encoded', 'car_model_encoded']

# Add one-hot encoded columns
for col in df_encoded.columns:
    if col.startswith(('transmission_', 'body_type_', 'fuel_type_', 'Age_Category_')):
        feature_cols.append(col)

print(f"Total features: {len(feature_cols)}")

# Prepare training data
X = df_encoded[feature_cols].fillna(0)
y = df_encoded['price'] / 1000000  # Scale target to millions

print("Training model...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

# Create scaler (though not used in prediction, keeping for compatibility)
scaler = StandardScaler()
scaler.fit(X_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\nSaving model files...")

# Save all required files
try:
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le_brand, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # Save additional metadata for debugging
    metadata = {
        'sklearn_version': '1.5.1',
        'feature_count': len(X.columns),
        'training_samples': len(X_train),
        'test_score': test_score,
        'feature_names': X.columns.tolist()
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✅ All files saved successfully!")
    print(f"✅ Model R² Score: {test_score:.4f}")
    print(f"✅ Features saved: {len(X.columns)}")
    
except Exception as e:
    print(f"❌ Error saving files: {e}")

print("\nModel training completed!")