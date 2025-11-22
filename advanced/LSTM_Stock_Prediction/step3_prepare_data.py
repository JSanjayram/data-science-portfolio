# 🔹 Step 3 — Prepare the data for the LSTM
# We'll use the past 60 days to predict the next day.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('aapl_data.csv', index_col=0, parse_dates=True)

# Extract closing prices
close_prices = df["Close"].values.reshape(-1, 1)

# Scale prices between 0 and 1 (helps neural networks learn)
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(close_prices)

lookback = 60   # number of days to look back
X, y = [], []

for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])  # last 60 days
    y.append(scaled[i, 0])             # today's price

X, y = np.array(X), np.array(y)

# Reshape input to be [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train/test sets (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {y_train.shape}")

# Save processed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save scaler for later use
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("💾 Processed data saved!")