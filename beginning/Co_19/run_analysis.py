#!/usr/bin/env python3
"""
Quick runner for COVID-19 analysis
"""

import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True

def run_analysis():
    try:
        subprocess.check_call([sys.executable, "covid19_analysis.py"])
        print("✅ COVID-19 analysis completed!")
    except subprocess.CalledProcessError:
        print("❌ Analysis failed")
        return False
    return True

if __name__ == "__main__":
    print("🦠 Starting COVID-19 Trends Analysis...")
    
    if install_requirements():
        run_analysis()
    
    print("🎉 Done! Check PNG files for visualizations.")