# 🔹 Step 5 — Make predictions

import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and data
model = load_model('lstm_model.h5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("🔮 Making predictions...")
pred = model.predict(X_test)

# Undo scaling to get actual dollar values
pred_prices = scaler.inverse_transform(pred)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

print(f"📊 Predictions shape: {pred_prices.shape}")
print(f"📊 Actual prices shape: {actual_prices.shape}")

# Save predictions
np.save('pred_prices.npy', pred_prices)
np.save('actual_prices.npy', actual_prices)

print("💾 Predictions saved!")
print(f"📈 Sample predictions:")
for i in range(5):
    print(f"Day {i+1}: Actual=${actual_prices[i][0]:.2f}, Predicted=${pred_prices[i][0]:.2f}")

print("✅ Predictions completed!")