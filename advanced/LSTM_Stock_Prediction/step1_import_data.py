# 🔹 Step 1 — Import packages and download data

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download Apple's (AAPL) daily data from 2016–2024
print("📊 Downloading AAPL stock data...")
df = yf.download("AAPL", start="2016-01-01", end="2024-01-01")

# Look at the first few rows
print("\n📋 First 5 rows of data:")
print(df.head())

print(f"\n📈 Data shape: {df.shape}")
print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Save data for next steps
df.to_csv('aapl_data.csv')
print("💾 Data saved to 'aapl_data.csv'")