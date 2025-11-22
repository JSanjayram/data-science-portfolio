# 🧠 Stock Price Prediction using LSTM (Complete Master File)

import subprocess
import sys
import os

# =============================================================================
# 🔹 Step 0 — Install required libraries
# =============================================================================

def step0_install():
    print("🔹 Step 0 — Install required libraries")
    print("="*50)
    
    packages = [
        'yfinance==0.1.87',
        'tensorflow==2.8.4', 
        'pandas==1.5.3',
        'scikit-learn==1.1.3',
        'matplotlib==3.5.3',
        'numpy==1.21.6'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed successfully!\n")

# =============================================================================
# 🔹 Step 1 — Import packages and download data
# =============================================================================

def step1_import_data():
    print("🔹 Step 1 — Import packages and download data")
    print("="*50)
    
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
    
    return df

# =============================================================================
# 🔹 Step 2 — Visualize the closing price
# =============================================================================

def step2_visualize(df):
    print("\n🔹 Step 2 — Visualize the closing price")
    print("="*50)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,4))
    plt.plot(df["Close"])
    plt.title("Apple Closing Price (2016–2024)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"📊 Current Price: ${df['Close'][-1]:.2f}")
    print(f"📈 Highest Price: ${df['Close'].max():.2f}")
    print(f"📉 Lowest Price: ${df['Close'].min():.2f}")
    print(f"📊 Average Price: ${df['Close'].mean():.2f}")

# =============================================================================
# 🔹 Step 3 — Prepare the data for the LSTM
# =============================================================================

def step3_prepare_data(df):
    print("\n🔹 Step 3 — Prepare the data for the LSTM")
    print("="*50)
    
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
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
    
    return X_train, X_test, y_train, y_test, scaler

# =============================================================================
# 🔹 Step 4 — Build and train the LSTM model
# =============================================================================

def step4_build_train(X_train, y_train):
    print("\n🔹 Step 4 — Build and train the LSTM model")
    print("="*50)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import matplotlib.pyplot as plt
    
    print("🤖 Building LSTM model...")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    print("📋 Model Summary:")
    model.summary()

    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("✅ Model training completed!")
    return model

# =============================================================================
# 🔹 Step 5 — Make predictions
# =============================================================================

def step5_predictions(model, X_test, y_test, scaler):
    print("\n🔹 Step 5 — Make predictions")
    print("="*50)
    
    import numpy as np
    
    print("🔮 Making predictions...")
    pred = model.predict(X_test)

    # Undo scaling to get actual dollar values
    pred_prices = scaler.inverse_transform(pred)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

    print(f"📊 Predictions shape: {pred_prices.shape}")
    print(f"📊 Actual prices shape: {actual_prices.shape}")

    print(f"📈 Sample predictions:")
    for i in range(5):
        print(f"Day {i+1}: Actual=${actual_prices[i][0]:.2f}, Predicted=${pred_prices[i][0]:.2f}")

    print("✅ Predictions completed!")
    return pred_prices, actual_prices

# =============================================================================
# 🔹 Step 6 — Plot actual vs predicted prices
# =============================================================================

def step6_plot_results(pred_prices, actual_prices):
    print("\n🔹 Step 6 — Plot actual vs predicted prices")
    print("="*50)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10,4))
    plt.plot(actual_prices, label='Actual', color='blue')
    plt.plot(pred_prices, label='Predicted', color='red')
    plt.title("Predicted vs Actual Apple Prices")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate accuracy percentage
    accuracy = 100 - np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    print(f"📊 Model Accuracy: {accuracy:.2f}%")

    # Show some statistics
    print(f"\n📈 Actual Price Stats:")
    print(f"   Mean: ${np.mean(actual_prices):.2f}")
    print(f"   Max: ${np.max(actual_prices):.2f}")
    print(f"   Min: ${np.min(actual_prices):.2f}")

    print(f"\n🔮 Predicted Price Stats:")
    print(f"   Mean: ${np.mean(pred_prices):.2f}")
    print(f"   Max: ${np.max(pred_prices):.2f}")
    print(f"   Min: ${np.min(pred_prices):.2f}")

    print("✅ Visualization completed!")

# =============================================================================
# 🔹 Step 7 — Evaluate with RMSE (error)
# =============================================================================

def step7_evaluate(pred_prices, actual_prices):
    print("\n🔹 Step 7 — Evaluate with RMSE (error)")
    print("="*50)
    
    from sklearn.metrics import mean_squared_error
    import math
    import numpy as np
    
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actual_prices, pred_prices))
    print("📊 Root Mean Square Error (RMSE):", f"${rmse:.2f}")

    # Calculate additional metrics
    mae = np.mean(np.abs(actual_prices - pred_prices))
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

    print(f"📊 Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Performance interpretation
    if rmse < 5:
        performance = "Excellent"
    elif rmse < 10:
        performance = "Good"
    elif rmse < 20:
        performance = "Fair"
    else:
        performance = "Needs Improvement"

    print(f"\n🎯 Model Performance: {performance}")
    print(f"💡 Lower RMSE = Better predictions")

    # Show prediction vs actual comparison
    print(f"\n📋 Last 10 Predictions vs Actual:")
    for i in range(-10, 0):
        diff = pred_prices[i][0] - actual_prices[i][0]
        print(f"Day {i+10}: Actual=${actual_prices[i][0]:.2f}, Predicted=${pred_prices[i][0]:.2f}, Diff=${diff:.2f}")

    print("✅ Evaluation completed!")

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def main():
    print("🧠 Stock Price Prediction using LSTM (Complete Pipeline)")
    print("🔹 All 7 Steps in One Master File")
    print("="*60)
    
    try:
        # Step 0: Install packages
        step0_install()
        
        # Step 1: Import and download data
        df = step1_import_data()
        
        # Step 2: Visualize data
        step2_visualize(df)
        
        # Step 3: Prepare data
        X_train, X_test, y_train, y_test, scaler = step3_prepare_data(df)
        
        # Step 4: Build and train model
        model = step4_build_train(X_train, y_train)
        
        # Step 5: Make predictions
        pred_prices, actual_prices = step5_predictions(model, X_test, y_test, scaler)
        
        # Step 6: Plot results
        step6_plot_results(pred_prices, actual_prices)
        
        # Step 7: Evaluate model
        step7_evaluate(pred_prices, actual_prices)
        
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("💡 Make sure all packages are installed correctly")

if __name__ == "__main__":
    main()