# 🧠 Stock Price Prediction using LSTM (Beginner Friendly)

A complete step-by-step implementation of LSTM neural networks for stock price prediction, designed for beginners to understand each component of the process.

## 📁 Project Structure

```
LSTM_Stock_Prediction/
├── step0_install.py          # Install required libraries
├── step1_import_data.py      # Import packages and download data
├── step2_visualize.py        # Visualize the closing price
├── step3_prepare_data.py     # Prepare data for LSTM
├── step4_build_train.py      # Build and train LSTM model
├── step5_predictions.py      # Make predictions
├── step6_plot_results.py     # Plot actual vs predicted prices
├── step7_evaluate.py         # Evaluate with RMSE
├── run_all_steps.py          # Run complete pipeline
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python run_all_steps.py
```

### Option 2: Run Individual Steps
```bash
python step0_install.py      # Install packages
python step1_import_data.py  # Download data
python step2_visualize.py    # Visualize data
python step3_prepare_data.py # Prepare for LSTM
python step4_build_train.py  # Train model
python step5_predictions.py  # Make predictions
python step6_plot_results.py # Plot results
python step7_evaluate.py     # Evaluate performance
```

## 📋 Step-by-Step Breakdown

### 🔹 Step 0 — Install Required Libraries
- Installs Python 3.8 compatible versions
- Packages: yfinance, tensorflow, pandas, scikit-learn, matplotlib

### 🔹 Step 1 — Import Packages and Download Data
- Downloads Apple (AAPL) stock data from 2016-2024
- Displays first 5 rows and basic statistics
- Saves data to CSV for next steps

### 🔹 Step 2 — Visualize the Closing Price
- Creates a line plot of Apple's closing price over time
- Shows price trends and patterns
- Displays key statistics (current, highest, lowest, average prices)

### 🔹 Step 3 — Prepare Data for LSTM
- Extracts closing prices and scales them (0-1 range)
- Creates sequences of 60 days to predict next day
- Splits data into 80% training, 20% testing
- Saves processed data for model training

### 🔹 Step 4 — Build and Train LSTM Model
- Creates Sequential model with 2 LSTM layers (50 units each)
- Compiles with Adam optimizer and MSE loss
- Trains for 20 epochs with validation split
- Saves trained model and plots training history

### 🔹 Step 5 — Make Predictions
- Loads trained model and test data
- Makes predictions on test set
- Converts scaled predictions back to actual dollar values
- Saves predictions for evaluation

### 🔹 Step 6 — Plot Actual vs Predicted Prices
- Creates comparison plot of actual vs predicted prices
- Calculates and displays model accuracy percentage
- Shows statistical comparison between actual and predicted values

### 🔹 Step 7 — Evaluate with RMSE
- Calculates Root Mean Square Error (RMSE)
- Computes additional metrics (MAE, MAPE)
- Provides performance interpretation
- Shows detailed prediction vs actual comparison

## 📊 Expected Results

- **RMSE**: Typically $2-8 for AAPL predictions
- **Accuracy**: Usually 85-95% for short-term predictions
- **Training Time**: 2-5 minutes on modern hardware

## 🎯 Learning Objectives

After completing this project, you'll understand:
- How to download and preprocess financial data
- LSTM neural network architecture for time series
- Data scaling and sequence creation for deep learning
- Model training, validation, and evaluation
- Performance metrics for regression problems

## ⚠️ Important Notes

### Educational Purpose
- This project is for learning machine learning concepts
- **NOT for actual trading decisions**
- Stock markets are unpredictable and risky

### Technical Limitations
- Model uses only historical price data
- Doesn't account for external factors (news, events)
- Past performance doesn't guarantee future results

## 🔧 Troubleshooting

### Common Issues:
1. **Package Installation**: Use Python 3.8 compatible versions
2. **Data Download**: Check internet connection for yfinance
3. **Memory Issues**: Reduce batch size or sequence length
4. **Training Time**: Reduce epochs for faster training

### Python 3.8 Compatibility:
- Uses yfinance==0.1.87 (compatible version)
- TensorFlow 2.8.4 works with Python 3.8
- All packages tested for compatibility

## 📈 Extensions

### Beginner Improvements:
- Try different stocks (GOOGL, MSFT, TSLA)
- Adjust lookback period (30, 90, 120 days)
- Experiment with different LSTM units

### Advanced Features:
- Add volume and technical indicators
- Implement multiple stock prediction
- Use ensemble methods
- Add sentiment analysis from news

## 🤝 Contributing

1. Fork the repository
2. Try different stocks and parameters
3. Share your results and improvements
4. Help others learn in discussions

## 📞 Support

- Check individual step files for detailed comments
- Review error messages for troubleshooting hints
- Experiment with different parameters to understand impact

---

**Built for educational purposes - Learn responsibly! 📚**