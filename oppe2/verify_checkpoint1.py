import os
import pandas as pd
import joblib

def verify_checkpoint1():
    print("=== CHECKPOINT 1 VERIFICATION ===")
    
    # Check data files
    print("\n1. Data Files:")
    try:
        features = pd.read_csv("data/wine_features.csv")
        targets = pd.read_csv("data/wine_targets.csv")
        print(f"   ✅ Features shape: {features.shape}")
        print(f"   ✅ Targets shape: {targets.shape}")
        print(f"   ✅ Location column added: {'location' in features.columns}")
    except Exception as e:
        print(f"   ❌ Data files error: {e}")
    
    # Check models
    print("\n2. Trained Models:")
    model_files = ["random_forest_model.pkl", "logistic_regression_model.pkl"]
    for model_file in model_files:
        try:
            model_path = f"models/{model_file}"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"   ✅ {model_file} loaded successfully")
            else:
                print(f"   ❌ {model_file} not found")
        except Exception as e:
            print(f"   ❌ {model_file} error: {e}")
    
    # Check MLflow runs
    print("\n3. MLflow Runs:")
    try:
        mlrun_dirs = [d for d in os.listdir("mlruns") if d.isdigit()]
        if mlrun_dirs:
            print(f"   ✅ MLflow experiments found: {len(mlrun_dirs)}")
        else:
            print("   ❌ No MLflow experiments found")
    except:
        print("   ❌ MLflow directory not accessible")
    
    # Check summary
    print("\n4. Training Summary:")
    try:
        if os.path.exists("models/training_summary.txt"):
            with open("models/training_summary.txt", "r") as f:
                print("   ✅ Summary file exists:")
                print("   " + f.read().replace("\n", "\n   "))
        else:
            print("   ❌ Training summary not found")
    except Exception as e:
        print(f"   ❌ Summary error: {e}")
    
    print("\n=== CHECKPOINT 1 STATUS ===")
    return True

if __name__ == "__main__":
    verify_checkpoint1()
