# 🔹 Step 6 — Plot actual vs predicted prices

import numpy as np
import matplotlib.pyplot as plt

# Load predictions
pred_prices = np.load('pred_prices.npy')
actual_prices = np.load('actual_prices.npy')

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