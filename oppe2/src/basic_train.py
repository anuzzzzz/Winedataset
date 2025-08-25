import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def setup_data():
    """Load and prepare wine dataset"""
    print("Loading wine dataset...")
    
    # Load wine dataset
    wine_data = load_wine()
    X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    y = wine_data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Add synthetic location attribute (sensitive feature)
    np.random.seed(42)
    location = np.random.choice([0, 1], size=len(X))
    X['location'] = location
    
    print(f"Added location attribute. Final shape: {X.shape}")
    
    # Save raw data
    os.makedirs("data", exist_ok=True)
    X.to_csv("data/wine_features.csv", index=False)
    pd.DataFrame(y, columns=['target']).to_csv("data/wine_targets.csv", index=False)
    
    print("Data saved to data/ directory")
    return X, y

def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train model and log results"""
    print(f"\nTraining {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("model_type", model_name)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)
        
        return accuracy, model

def main():
    print("=== Wine Classification - Basic Training ===")
    
    # Setup MLflow
    mlflow.set_experiment("wine-classification-basic")
    print("MLflow experiment set up")
    
    # Load and prepare data
    X, y = setup_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train different models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = ""
    
    for name, model in models.items():
        accuracy, trained_model = train_and_evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results[name] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    print(f"\n=== RESULTS SUMMARY ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    
    print(f"\nBest Model: {best_model_name} ({best_accuracy:.4f})")
    
    # Save summary
    with open("models/training_summary.txt", "w") as f:
        f.write("Training Summary\n")
        f.write("================\n\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")
        f.write(f"\nBest Model: {best_model_name} ({best_accuracy:.4f})\n")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
