#!/usr/bin/env python3
"""
Quick runner for BMI Calculator Web App
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

def run_streamlit_app():
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "bmi_app.py"])
        print("✅ BMI Calculator app launched!")
    except subprocess.CalledProcessError:
        print("❌ Failed to launch app")
        return False
    return True

if __name__ == "__main__":
    print("⚖️ Starting BMI Calculator Web App...")
    
    if install_requirements():
        print("🚀 Launching Streamlit app...")
        run_streamlit_app()
    
    print("🎉 App should open in your browser!")