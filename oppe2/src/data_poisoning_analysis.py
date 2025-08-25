import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.report import Report
from evidently.metric_suite import DataQualityReport
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
import json
import os

def add_poison(X, y, poison_rate, poison_type='noise'):
    """Add different types of poisoning to the dataset"""
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    n_poison = int(len(X) * poison_rate)
    
    if n_poison == 0:
        return X_poisoned, y_poisoned
    
    # Select random samples to poison
    poison_indices = np.random.choice(len(X), n_poison, replace=False)
    
    if poison_type == 'noise':
        # Add Gaussian noise to features
        for idx in poison_indices:
            noise = np.random.normal(0, 0.5, len(X.columns))
            X_poisoned.iloc[idx] += noise
    
    elif poison_type == 'label_flip':
        # Flip labels randomly
        for idx in poison_indices:
            available_labels = [0, 1, 2]
            available_labels.remove(y_poisoned[idx])
            y_poisoned[idx] = np.random.choice(available_labels)
    
    elif poison_type == 'outlier':
        # Create extreme outliers
        for idx in poison_indices:
            # Set random features to extreme values
            n_features_to_poison = np.random.randint(1, min(5, len(X.columns)))
            features_to_poison = np.random.choice(X.columns[:-1], n_features_to_poison, replace=False)  # Exclude location
            
            for feature in features_to_poison:
                if np.random.random() > 0.5:
                    X_poisoned.loc[X_poisoned.index[idx], feature] *= 10  # Extreme high
                else:
                    X_poisoned.loc[X_poisoned.index[idx], feature] *= 0.1  # Extreme low
    
    return X_poisoned, y_poisoned

def analyze_poisoning_impact():
    """Analyze impact of different poisoning strategies"""
    print("🧪 STARTING DATA POISONING ANALYSIS")
    print("=" * 50)
    
    # Load original data
    wine_data = load_wine()
    X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    y = wine_data.target
    
    # Add synthetic location attribute
    np.random.seed(42)
    X['location'] = np.random.choice([0, 1], size=len(X))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Test different poisoning levels and types
    poison_levels = [0.0, 0.05, 0.10, 0.50]
    poison_types = ['noise', 'label_flip', 'outlier']
    
    results = {}
    
    for poison_type in poison_types:
        print(f"\n🔍 Testing {poison_type} poisoning...")
        results[poison_type] = {}
        
        for poison_rate in poison_levels:
            print(f"  📊 Poison level: {poison_rate*100}%")
            
            # Apply poisoning
            X_train_poisoned, y_train_poisoned = add_poison(X_train, y_train, poison_rate, poison_type)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_poisoned, y_train_poisoned)
            
            # Evaluate on clean test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[poison_type][f"poison_{int(poison_rate*100)}"] = {
                "accuracy": accuracy,
                "poison_rate": poison_rate,
                "poison_type": poison_type
            }
            
            print(f"    Accuracy: {accuracy:.4f}")
            
            # Generate Evidently report for drift detection
            if poison_rate > 0:
                try:
                    reference_data = X_train.sample(min(100, len(X_train)), random_state=42)
                    current_data = X_train_poisoned.sample(min(100, len(X_train_poisoned)), random_state=42)
                    
                    report = Report(metrics=[
                        DatasetDriftMetric(),
                        DatasetMissingValuesMetric()
                    ])
                    
                    report.run(reference_data=reference_data, current_data=current_data)
                    
                    # Save report
                    os.makedirs("reports", exist_ok=True)
                    report_name = f"reports/evidently_{poison_type}_{int(poison_rate*100)}.html"
                    report.save_html(report_name)
                    
                except Exception as e:
                    print(f"    Evidently report failed: {e}")
    
    # Save all results
    with open("reports/poisoning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_poisoning_visualization(results)
    
    # Generate insights
    generate_poisoning_insights(results)
    
    print(f"\n✅ Data poisoning analysis completed!")
    print(f"📁 Results saved to: reports/poisoning_results.json")

def create_poisoning_visualization(results):
    """Create visualization of poisoning impact"""
    
    plt.figure(figsize=(15, 5))
    
    for i, poison_type in enumerate(['noise', 'label_flip', 'outlier']):
        plt.subplot(1, 3, i+1)
        
        poison_levels = []
        accuracies = []
        
        for key, result in results[poison_type].items():
            poison_levels.append(result['poison_rate'] * 100)
            accuracies.append(result['accuracy'])
        
        plt.plot(poison_levels, accuracies, 'o-', linewidth=2, markersize=8)
        plt.title(f'{poison_type.replace("_", " ").title()} Poisoning')
        plt.xlabel('Poison Level (%)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add values on points
        for x, y in zip(poison_levels, accuracies):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('reports/poisoning_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_poisoning_insights(results):
    """Generate insights and mitigation strategies"""
    
    insights = []
    
    # Analyze robustness across poison types
    for poison_type in results.keys():
        baseline_acc = results[poison_type]['poison_0']['accuracy']
        severe_acc = results[poison_type]['poison_50']['accuracy']
        degradation = (baseline_acc - severe_acc) / baseline_acc * 100
        
        insights.append(f"• {poison_type.replace('_', ' ').title()} Poisoning:")
        insights.append(f"  - Baseline accuracy: {baseline_acc:.4f}")
        insights.append(f"  - 50% poison accuracy: {severe_acc:.4f}")
        insights.append(f"  - Performance degradation: {degradation:.1f}%")
        insights.append("")
    
    # Mitigation strategies
    mitigations = [
        "MITIGATION STRATEGIES:",
        "",
        "1. Detection Methods:",
        "   • Statistical monitoring of feature distributions",
        "   • Anomaly detection in training data",
        "   • Cross-validation with holdout sets",
        "   • Evidently reports for data drift detection",
        "",
        "2. Prevention Strategies:",
        "   • Input validation and sanitization",
        "   • Robust training techniques (e.g., robust statistics)",
        "   • Ensemble methods for increased resilience",
        "   • Regular model retraining with clean data",
        "",
        "3. Response Procedures:",
        "   • Immediate: Rollback to previous model version",
        "   • Short-term: Retrain with cleaned dataset",
        "   • Long-term: Implement better data governance",
        "",
        "4. Monitoring:",
        "   • Continuous performance monitoring",
        "   • Alert systems for accuracy drops",
        "   • Regular data quality assessments"
    ]
    
    # Save insights
    with open("reports/poisoning_insights.txt", "w") as f:
        f.write("DATA POISONING ANALYSIS - INSIGHTS & MITIGATIONS\n")
        f.write("=" * 55 + "\n\n")
        f.write("ROBUSTNESS ANALYSIS:\n\n")
        f.write("\n".join(insights))
        f.write("\n" + "\n".join(mitigations))
    
    print("📋 Insights and mitigation strategies saved to: reports/poisoning_insights.txt")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    analyze_poisoning_impact()
