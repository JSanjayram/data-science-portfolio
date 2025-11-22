# 🔹 Step 2 — Visualize the closing price

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('aapl_data.csv', index_col=0, parse_dates=True)

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