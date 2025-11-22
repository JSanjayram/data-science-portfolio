# 🧠 Stock Price Prediction using LSTM (Complete Pipeline)

import subprocess
import sys
import os

def run_step(step_file, step_name):
    """Run a step and handle any errors"""
    print(f"\n{'='*50}")
    print(f"🚀 Running {step_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, step_file], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"❌ Error in {step_name}:")
            print(result.stderr)
            return False
        
        print(f"✅ {step_name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to run {step_name}: {e}")
        return False

def main():
    print("🧠 Stock Price Prediction using LSTM (Beginner Friendly)")
    print("🔹 Complete 7-Step Pipeline")
    
    steps = [
        ("step0_install.py", "Step 0 — Install required libraries"),
        ("step1_import_data.py", "Step 1 — Import packages and download data"),
        ("step2_visualize.py", "Step 2 — Visualize the closing price"),
        ("step3_prepare_data.py", "Step 3 — Prepare the data for LSTM"),
        ("step4_build_train.py", "Step 4 — Build and train LSTM model"),
        ("step5_predictions.py", "Step 5 — Make predictions"),
        ("step6_plot_results.py", "Step 6 — Plot actual vs predicted prices"),
        ("step7_evaluate.py", "Step 7 — Evaluate with RMSE")
    ]
    
    print(f"\n📋 Pipeline Overview:")
    for i, (_, step_name) in enumerate(steps):
        print(f"   {i}: {step_name}")
    
    # Ask user which steps to run
    print(f"\n🎯 Options:")
    print("1. Run all steps (0-7)")
    print("2. Run specific step")
    print("3. Run from specific step onwards")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Run all steps
        for step_file, step_name in steps:
            if not run_step(step_file, step_name):
                print(f"\n❌ Pipeline stopped at {step_name}")
                return
        print(f"\n🎉 All steps completed successfully!")
        
    elif choice == "2":
        # Run specific step
        step_num = int(input("Enter step number (0-7): "))
        if 0 <= step_num < len(steps):
            step_file, step_name = steps[step_num]
            run_step(step_file, step_name)
        else:
            print("❌ Invalid step number")
            
    elif choice == "3":
        # Run from specific step onwards
        start_step = int(input("Enter starting step number (0-7): "))
        if 0 <= start_step < len(steps):
            for step_file, step_name in steps[start_step:]:
                if not run_step(step_file, step_name):
                    print(f"\n❌ Pipeline stopped at {step_name}")
                    return
            print(f"\n🎉 Pipeline completed from step {start_step}!")
        else:
            print("❌ Invalid step number")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()