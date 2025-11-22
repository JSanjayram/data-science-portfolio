# 🔹 Step 0 — Install required libraries
# Run this first:

import subprocess
import sys

def install_packages():
    packages = [
        'yfinance==0.1.87',  # Python 3.8 compatible version
        'tensorflow==2.8.4',
        'pandas==1.5.3',
        'scikit-learn==1.1.3',
        'matplotlib==3.5.3',
        'numpy==1.21.6'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed successfully!")

if __name__ == "__main__":
    install_packages()