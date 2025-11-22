#!/usr/bin/env python3
"""
Quick runner script for Titanic EDA
"""

import subprocess
import sys

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True

def run_analysis():
    """Run the main analysis"""
    try:
        subprocess.check_call([sys.executable, "titanic_eda.py"])
        print("✅ Analysis completed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Analysis failed")
        return False
    return True

if __name__ == "__main__":
    print("🚢 Starting Titanic EDA Analysis...")
    
    if install_requirements():
        run_analysis()
    
    print("🎉 Done! Check titanic_analysis.png for visualizations.")