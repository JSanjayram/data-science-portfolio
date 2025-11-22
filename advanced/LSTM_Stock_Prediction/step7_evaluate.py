# 🔹 Step 7 — Evaluate with RMSE (error)

import numpy as np
from sklearn.metrics import mean_squared_error
import math

# Load predictions
pred_prices = np.load('pred_prices.npy')
actual_prices = np.load('actual_prices.npy')

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