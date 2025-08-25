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
import json
import os
from sklearn.preprocessing import StandardScaler

def add_poison(X, y, poison_rate, poison_type='noise'):
    """Add different types of poisoning to the dataset"""
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    n_poison = int(len(X) * poison_rate)
    
    if n_poison == 0:
        return X_poisoned, y_poisoned
    
    # Select random samples to poison
    np.random.seed(42)  # For reproducible results
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
    
    print(f"📊 Dataset overview:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(X.columns)} (including location)")
    print(f"   Classes: {np.bincount(y)} (Cultivar 0, 1, 2)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Test different poisoning levels and types
    poison_levels = [0.0, 0.05, 0.10, 0.50]
    poison_types = ['noise', 'label_flip', 'outlier']
    
    results = {}
    detailed_results = []
    
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
            
            # Calculate additional metrics
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            
            result_entry = {
                "poison_type": poison_type,
                "poison_rate": poison_rate,
                "accuracy": accuracy,
                "class_accuracies": class_accuracies.tolist(),
                "confusion_matrix": conf_matrix.tolist()
            }
            
            results[poison_type][f"poison_{int(poison_rate*100)}"] = {
                "accuracy": accuracy,
                "poison_rate": poison_rate,
                "poison_type": poison_type
            }
            
            detailed_results.append(result_entry)
            
            print(f"    Overall Accuracy: {accuracy:.4f}")
            print(f"    Class Accuracies: {[f'{acc:.3f}' for acc in class_accuracies]}")
            
            # Calculate data drift metrics (simple statistical approach)
            if poison_rate > 0:
                calculate_simple_drift(X_train, X_train_poisoned, poison_type, poison_rate)
    
    # Save all results
    with open("reports/poisoning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("reports/poisoning_detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create visualizations
    create_poisoning_visualization(results)
    
    # Generate insights
    generate_poisoning_insights(results, detailed_results)
    
    print(f"\n✅ Data poisoning analysis completed!")
    print(f"📁 Results saved to: reports/poisoning_results.json")

def calculate_simple_drift(X_original, X_poisoned, poison_type, poison_rate):
    """Calculate simple drift metrics without Evidently"""
    
    # Calculate feature-wise statistical differences
    drift_metrics = {}
    
    for feature in X_original.columns:
        orig_mean = X_original[feature].mean()
        poison_mean = X_poisoned[feature].mean()
        orig_std = X_original[feature].std()
        poison_std = X_poisoned[feature].std()
        
        # Mean shift
        mean_shift = abs(poison_mean - orig_mean) / orig_std if orig_std > 0 else 0
        
        # Standard deviation change
        std_change = abs(poison_std - orig_std) / orig_std if orig_std > 0 else 0
        
        drift_metrics[feature] = {
            "mean_shift": mean_shift,
            "std_change": std_change
        }
    
    # Save drift analysis
    drift_summary = {
        "poison_type": poison_type,
        "poison_rate": poison_rate,
        "avg_mean_shift": np.mean([m["mean_shift"] for m in drift_metrics.values()]),
        "avg_std_change": np.mean([m["std_change"] for m in drift_metrics.values()]),
        "max_mean_shift": max([m["mean_shift"] for m in drift_metrics.values()]),
        "feature_drift": drift_metrics
    }
    
    # Save individual drift report
    drift_filename = f"reports/drift_analysis_{poison_type}_{int(poison_rate*100)}.json"
    with open(drift_filename, "w") as f:
        json.dump(drift_summary, f, indent=2)

def create_poisoning_visualization(results):
    """Create visualization of poisoning impact"""
    
    plt.figure(figsize=(15, 5))
    
    colors = ['blue', 'red', 'green']
    
    for i, poison_type in enumerate(['noise', 'label_flip', 'outlier']):
        plt.subplot(1, 3, i+1)
        
        poison_levels = []
        accuracies = []
        
        for key, result in results[poison_type].items():
            poison_levels.append(result['poison_rate'] * 100)
            accuracies.append(result['accuracy'])
        
        plt.plot(poison_levels, accuracies, 'o-', linewidth=2, markersize=8, 
                color=colors[i], label=f'{poison_type.replace("_", " ").title()}')
        plt.title(f'{poison_type.replace("_", " ").title()} Poisoning Impact')
        plt.xlabel('Poison Level (%)')
        plt.ylabel('Model Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add values on points
        for x, y in zip(poison_levels, accuracies):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/poisoning_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    poison_levels = [0, 5, 10, 50]
    
    for i, poison_type in enumerate(['noise', 'label_flip', 'outlier']):
        accuracies = []
        for level in poison_levels:
            key = f"poison_{level}"
            accuracies.append(results[poison_type][key]['accuracy'])
        
        plt.plot(poison_levels, accuracies, 'o-', linewidth=3, markersize=10, 
                label=f'{poison_type.replace("_", " ").title()} Attack', color=colors[i])
    
    plt.title('Model Robustness Under Different Poisoning Attacks', fontsize=16, fontweight='bold')
    plt.xlabel('Poison Level (%)', fontsize=14)
    plt.ylabel('Model Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim([0, 1.05])
    
    # Add baseline reference
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
    
    plt.tight_layout()
    plt.savefig('reports/poisoning_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Visualizations created:")
    print("   • reports/poisoning_impact.png")
    print("   • reports/poisoning_comparison.png")

def generate_poisoning_insights(results, detailed_results):
    """Generate insights and mitigation strategies"""
    
    insights = []
    insights.append("DATA POISONING ROBUSTNESS ANALYSIS")
    insights.append("=" * 45)
    insights.append("")
    
    # Analyze robustness across poison types
    for poison_type in results.keys():
        baseline_acc = results[poison_type]['poison_0']['accuracy']
        moderate_acc = results[poison_type]['poison_10']['accuracy'] 
        severe_acc = results[poison_type]['poison_50']['accuracy']
        moderate_degradation = (baseline_acc - moderate_acc) / baseline_acc * 100
        severe_degradation = (baseline_acc - severe_acc) / baseline_acc * 100
        
        insights.append(f"🔍 {poison_type.replace('_', ' ').title()} Poisoning Analysis:")
        insights.append(f"   • Baseline accuracy: {baseline_acc:.4f}")
        insights.append(f"   • 10% poison accuracy: {moderate_acc:.4f} (-{moderate_degradation:.1f}%)")
        insights.append(f"   • 50% poison accuracy: {severe_acc:.4f} (-{severe_degradation:.1f}%)")
        
        if severe_degradation < 10:
            insights.append(f"   ✅ ROBUST: Model shows good resistance to {poison_type} attacks")
        elif severe_degradation < 30:
            insights.append(f"   ⚠️  MODERATE: Some degradation under {poison_type} attacks")
        else:
            insights.append(f"   ❌ VULNERABLE: Significant degradation under {poison_type} attacks")
        
        insights.append("")
    
    # Overall assessment
    insights.extend([
        "🎯 OVERALL ASSESSMENT:",
        "",
        "The Random Forest model demonstrates varying levels of robustness",
        "against different types of data poisoning attacks:",
        "",
        "• NOISE ATTACKS: Model typically robust to feature noise",
        "• LABEL FLIPPING: More vulnerable to mislabeled training data", 
        "• OUTLIER INJECTION: Moderate impact from extreme values",
        "",
        "📋 KEY FINDINGS:",
        "",
        "1. Model maintains reasonable performance at 5-10% poison levels",
        "2. Severe degradation occurs at 50% poisoning (as expected)",
        "3. Label flipping attacks are typically most effective",
        "4. Feature noise has limited impact on tree-based models",
        "",
        "🛡️  MITIGATION STRATEGIES:",
        "",
        "DETECTION METHODS:",
        "• Statistical monitoring of feature distributions",
        "• Anomaly detection in training data pipeline",
        "• Cross-validation with holdout validation sets",
        "• Data quality checks and validation rules",
        "",
        "PREVENTION STRATEGIES:",  
        "• Input validation and sanitization at data ingestion",
        "• Robust training techniques (e.g., robust statistics)",
        "• Ensemble methods for increased resilience",
        "• Regular model retraining with verified clean data",
        "",
        "RESPONSE PROCEDURES:",
        "• Immediate: Rollback to previous model version",
        "• Short-term: Retrain with cleaned and verified dataset",
        "• Long-term: Implement stronger data governance framework",
        "",
        "MONITORING & ALERTING:",
        "• Continuous performance monitoring in production",
        "• Automated alerts for accuracy drops or drift detection",
        "• Regular data quality assessments and audits",
        "• Model explanation monitoring for decision consistency",
        "",
        "💡 RECOMMENDATIONS:",
        "",
        "1. Implement data validation pipelines before training",
        "2. Use ensemble methods to improve poison resistance",
        "3. Monitor model performance continuously in production", 
        "4. Establish baseline metrics for drift detection",
        "5. Create incident response procedures for data quality issues",
        "",
        "=" * 45
    ])
    
    # Save insights
    with open("reports/poisoning_insights.txt", "w") as f:
        f.write("\n".join(insights))
    
    print("📋 Insights and mitigation strategies saved to: reports/poisoning_insights.txt")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    analyze_poisoning_impact()
