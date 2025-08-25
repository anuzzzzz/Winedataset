import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame, selection_rate, equalized_odds_difference
from fairlearn.metrics import demographic_parity_difference, equalized_odds_ratio
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
from sklearn.ensemble import RandomForestClassifier
import json
import os

def comprehensive_fairness_analysis():
    """Perform comprehensive fairness analysis"""
    print("⚖️  STARTING FAIRNESS ANALYSIS")
    print("=" * 50)
    
    # Load data
    wine_data = load_wine()
    X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    y = wine_data.target
    
    # Add synthetic location attribute (sensitive feature)
    np.random.seed(42)
    location = np.random.choice([0, 1], size=len(X), p=[0.6, 0.4])  # Imbalanced groups
    X['location'] = location
    
    print(f"📊 Dataset overview:")
    print(f"   Total samples: {len(X)}")
    print(f"   Location distribution: {np.bincount(location)}")
    print(f"   Classes: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features (excluding location)
    feature_cols = [col for col in X.columns if col != 'location']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    # Train baseline model
    print(f"\n🤖 Training baseline model...")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = baseline_model.predict(X_test_scaled)
    y_pred_proba = baseline_model.predict_proba(X_test_scaled)
    
    # Extract sensitive feature for test set
    sensitive_features_test = X_test['location']
    
    # Calculate fairness metrics
    print(f"\n📈 Calculating fairness metrics...")
    
    fairness_results = {}
    
    # Overall accuracy by group
    metric_frame = MetricFrame(
        metrics={
            'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred),
            'selection_rate': selection_rate
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features_test
    )
    
    print(f"\n📊 METRICS BY LOCATION GROUP:")
    print(metric_frame.by_group)
    
    # Calculate bias metrics
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features_test)
    
    fairness_results['demographic_parity_difference'] = float(dp_diff)
    fairness_results['equalized_odds_difference'] = float(eo_diff)
    fairness_results['metrics_by_group'] = metric_frame.by_group.to_dict()
    
    print(f"\n⚖️  FAIRNESS METRICS:")
    print(f"   Demographic Parity Difference: {dp_diff:.4f}")
    print(f"   Equalized Odds Difference: {eo_diff:.4f}")
    
    # Detailed analysis by class and group
    analyze_class_level_fairness(y_test, y_pred, sensitive_features_test, fairness_results)
    
    # Create visualizations
    create_fairness_visualizations(y_test, y_pred, y_pred_proba, sensitive_features_test)
    
    # Save results
    with open("reports/fairness_results.json", "w") as f:
        json.dump(fairness_results, f, indent=2)
    
    # Generate fairness report
    generate_fairness_report(fairness_results)
    
    print(f"\n✅ Fairness analysis completed!")
    return fairness_results

def analyze_class_level_fairness(y_true, y_pred, sensitive_features, results):
    """Analyze fairness at class level"""
    
    print(f"\n🔍 CLASS-LEVEL FAIRNESS ANALYSIS:")
    
    class_fairness = {}
    wine_classes = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
    
    for class_idx in range(3):
        print(f"\n   {wine_classes[class_idx]}:")
        
        # Binary predictions for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        # Calculate metrics for each group
        group_metrics = {}
        for group in [0, 1]:
            mask = sensitive_features == group
            if mask.sum() > 0:
                precision = np.mean(y_pred_binary[mask] == y_true_binary[mask])
                selection_rate = np.mean(y_pred_binary[mask])
                
                group_metrics[f'location_{group}'] = {
                    'accuracy': precision,
                    'selection_rate': selection_rate,
                    'sample_count': int(mask.sum())
                }
                
                print(f"     Location {group}: Acc={precision:.3f}, Selection={selection_rate:.3f}, N={mask.sum()}")
        
        class_fairness[wine_classes[class_idx]] = group_metrics
    
    results['class_level_fairness'] = class_fairness

def create_fairness_visualizations(y_true, y_pred, y_pred_proba, sensitive_features):
    """Create fairness visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction distribution by group
    ax1 = axes[0, 0]
    for group in [0, 1]:
        mask = sensitive_features == group
        pred_dist = np.bincount(y_pred[mask], minlength=3) / mask.sum()
        ax1.bar(np.arange(3) + group*0.35, pred_dist, width=0.35, 
                label=f'Location {group}', alpha=0.7)
    
    ax1.set_xlabel('Wine Cultivar')
    ax1.set_ylabel('Prediction Rate')
    ax1.set_title('Prediction Distribution by Location Group')
    ax1.set_xticks(np.arange(3) + 0.175)
    ax1.set_xticklabels(['Cultivar 0', 'Cultivar 1', 'Cultivar 2'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy by group and class
    ax2 = axes[0, 1]
    accuracy_data = []
    groups = []
    classes = []
    
    for class_idx in range(3):
        for group in [0, 1]:
            mask = (sensitive_features == group)
            if mask.sum() > 0:
                class_mask = (y_true == class_idx) & mask
                if class_mask.sum() > 0:
                    acc = np.mean(y_pred[class_mask] == y_true[class_mask])
                    accuracy_data.append(acc)
                    groups.append(f'Location {group}')
                    classes.append(f'Cultivar {class_idx}')
    
    # Create accuracy comparison
    df_acc = pd.DataFrame({
        'Accuracy': accuracy_data,
        'Group': groups,
        'Class': classes
    })
    
    if len(df_acc) > 0:
        pivot_acc = df_acc.pivot(index='Class', columns='Group', values='Accuracy')
        sns.heatmap(pivot_acc, annot=True, cmap='RdYlBu_r', ax=ax2, 
                   vmin=0, vmax=1, fmt='.3f')
        ax2.set_title('Accuracy by Group and Class')
    
    # 3. Confidence distribution by group
    ax3 = axes[1, 0]
    for group in [0, 1]:
        mask = sensitive_features == group
        max_probs = np.max(y_pred_proba[mask], axis=1)
        ax3.hist(max_probs, bins=20, alpha=0.6, label=f'Location {group}', density=True)
    
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Density')
    ax3.set_title('Prediction Confidence Distribution by Group')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion matrix comparison
    ax4 = axes[1, 1]
    from sklearn.metrics import confusion_matrix
    
    # Combine confusion matrices for both groups
    cm_combined = np.zeros((3, 3))
    for group in [0, 1]:
        mask = sensitive_features == group
        if mask.sum() > 0:
            cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1, 2])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_combined += cm_normalized * (mask.sum() / len(y_true))
    
    sns.heatmap(cm_combined, annot=True, fmt='.2f', cmap='Blues', ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Normalized Confusion Matrix (Combined)')
    ax4.set_xticklabels(['Cultivar 0', 'Cultivar 1', 'Cultivar 2'])
    ax4.set_yticklabels(['Cultivar 0', 'Cultivar 1', 'Cultivar 2'])
    
    plt.tight_layout()
    plt.savefig('reports/fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_fairness_report(results):
    """Generate comprehensive fairness report"""
    
    report_lines = [
        "FAIRNESS ANALYSIS REPORT",
        "=" * 50,
        "",
        "EXECUTIVE SUMMARY:",
        f"• Demographic Parity Difference: {results['demographic_parity_difference']:.4f}",
        f"• Equalized Odds Difference: {results['equalized_odds_difference']:.4f}",
        "",
        "INTERPRETATION:",
        "• Values close to 0 indicate fair treatment across groups",
        "• Demographic Parity measures selection rate equality",
        "• Equalized Odds measures accuracy equality across groups",
        "",
        "FINDINGS:",
    ]
    
    # Interpret the results
    dp_diff = results['demographic_parity_difference']
    eo_diff = results['equalized_odds_difference']
    
    if abs(dp_diff) < 0.1:
        report_lines.append("✅ GOOD: Low demographic parity difference suggests fair selection rates")
    else:
        report_lines.append("⚠️  CONCERN: High demographic parity difference indicates selection bias")
    
    if abs(eo_diff) < 0.1:
        report_lines.append("✅ GOOD: Low equalized odds difference suggests fair accuracy across groups")
    else:
        report_lines.append("⚠️  CONCERN: High equalized odds difference indicates accuracy bias")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "1. Continue monitoring fairness metrics in production",
        "2. Consider collecting more balanced training data",
        "3. Implement fairness constraints during training if bias detected",
        "4. Regular auditing of model decisions across demographic groups",
        "",
        "METHODOLOGY:",
        "• Synthetic location attribute used as sensitive feature",
        "• Standard fairness metrics from Fairlearn library",
        "• Analysis performed on held-out test set",
        "• Multi-class classification fairness assessment"
    ])
    
    with open("reports/fairness_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    print("📋 Fairness report saved to: reports/fairness_report.txt")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    comprehensive_fairness_analysis()
