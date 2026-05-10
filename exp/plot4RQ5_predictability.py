import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

from tools.config import (
    METHODS, DATASETS, MODELS, RANDOM_SEED_LIST, 
    pairs_to_analyze, considered_sensitive_attributes,
    METHODS_NAME, MODELS_NAME, DATASETS_NAME
)
from tools.utils import makedirs

# Setup output directory
script_dir = osp.dirname(osp.abspath(__file__))
results_dir = osp.join(script_dir, "results", "RQ5")
makedirs(results_dir)


def normalize_pair(pair):
    """Normalize metric pair to handle order invariance"""
    return tuple(sorted(pair))


def read_compatibility(method, dataset, classifier, seed):
    """Read compatibility JSON file"""
    compatibility_root = osp.join(
        script_dir, "../models", dataset, method, classifier, 
        f"seed_{seed}", "interactions", "compatibility", "compatibility.json"
    )
    try:
        with open(compatibility_root, 'r') as f:
            data = json.load(f)
        # Normalize all keys
        normalized_data = {}
        for key, value in data.items():
            metrics = key.split('-')
            normalized_key = '-'.join(sorted(metrics))
            normalized_data[normalized_key] = value
        return normalized_data
    except FileNotFoundError:
        return None


def compute_averaged_compatibility():
    """Step 4: Average compatibility across random seeds"""
    print("Step 4: Computing averaged compatibility...")
    
    averaged_compat = {}
    
    for method in METHODS:
        for dataset in DATASETS:
            for classifier in MODELS:
                key = (method, dataset, classifier)
                compat_list = []
                
                for seed in RANDOM_SEED_LIST:
                    compat = read_compatibility(method, dataset, classifier, seed)
                    if compat is not None:
                        compat_list.append(compat)
                
                if compat_list:
                    # Average across seeds
                    avg_compat = {}
                    for pair_key in compat_list[0].keys():
                        values = [c[pair_key] for c in compat_list if pair_key in c]
                        avg_compat[pair_key] = np.mean(values)
                    averaged_compat[key] = avg_compat
    
    return averaged_compat


def compute_omega_squared(data, factor_name):
    """Compute omega-squared (Type II ANOVA effect size)"""
    groups = data.groupby(factor_name)['compatibility'].apply(list)
    group_data = [np.array(g) for g in groups]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # Compute omega-squared
    n_groups = len(group_data)
    n_total = sum(len(g) for g in group_data)
    
    # Between-group sum of squares
    grand_mean = np.mean(np.concatenate(group_data))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
    
    # Within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in group_data)
    
    # Total sum of squares
    ss_total = ss_between + ss_within
    
    # Mean squares
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    
    # Omega-squared
    omega_sq = (ss_between - (n_groups - 1) * ms_within) / (ss_total + ms_within)
    omega_sq = max(0, omega_sq)  # Ensure non-negative
    
    return omega_sq


def analyze_factors_omega_squared(averaged_compat):
    """Step 4: Compute omega-squared for each factor and visualize (main effects only)"""
    print("Step 4: Analyzing factors with omega-squared (main effects)...")
    
    results = {}
    
    for pair in pairs_to_analyze:
        normalized_pair = normalize_pair(pair)
        pair_key = '-'.join(normalized_pair)
        
        # Collect data
        data_records = []
        for (method, dataset, classifier), compat_dict in averaged_compat.items():
            if pair_key in compat_dict:
                data_records.append({
                    'method': method,
                    'dataset': dataset,
                    'model': classifier,
                    'compatibility': compat_dict[pair_key]
                })
        
        df = pd.DataFrame(data_records)
        
        # Compute omega-squared for each factor
        omega_sq_dataset = compute_omega_squared(df, 'dataset')
        omega_sq_model = compute_omega_squared(df, 'model')
        omega_sq_method = compute_omega_squared(df, 'method')
        
        results[pair] = {
            'dataset': omega_sq_dataset,
            'model': omega_sq_model,
            'method': omega_sq_method
        }
    
    # Visualize
    n_pairs = len(pairs_to_analyze)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, pair in enumerate(pairs_to_analyze):
        ax = axes[idx]
        factors = ['Dataset', 'Model', 'Method']
        omega_values = [
            results[pair]['dataset'],
            results[pair]['model'],
            results[pair]['method']
        ]
        
        bars = ax.bar(factors, omega_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Omega-squared (ω²)', fontsize=12)
        ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(omega_values) * 1.2 if max(omega_values) > 0 else 0.5)
        
        # Annotate bars
        for bar, value in zip(bars, omega_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(osp.join(results_dir, 'omega_squared_main_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Omega-squared (main effects) analysis saved to {results_dir}/omega_squared_main_effects.png")
    return results


def compute_omega_squared_with_interactions(data):
    """Compute omega-squared including interaction effects using Type II ANOVA"""
    
    print(f"\nData shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Sample data:\n{data.head()}")
    
    data = data.copy()
    data['dataset'] = data['dataset'].astype(str)
    data['model'] = data['model'].astype(str)
    data['method'] = data['method'].astype(str)
    data['compatibility'] = pd.to_numeric(data['compatibility'], errors='coerce')
    
    data = data.dropna(subset=['compatibility'])
    
    if len(data) == 0:
        print("Error: No valid data after cleaning")
        return {}, 0, None, 0, 0
    
    # Fit full model with two-way interactions
    formula = ('compatibility ~ C(dataset) + C(model) + C(method) + '
               'C(dataset):C(model) + C(dataset):C(method) + C(model):C(method)')
    
    try:
        model = ols(formula, data=data).fit()
        
        # Type II ANOVA
        anova_table = anova_lm(model, typ=2)
        
        print(f"\nANOVA Table:\n{anova_table}")
        print(f"ANOVA columns: {anova_table.columns.tolist()}")
        
        anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
        
        # Compute omega-squared for each term
        ss_total = anova_table['sum_sq'].sum()
        ms_residual = anova_table.loc['Residual', 'mean_sq']
        
        omega_squared = {}
        for term in anova_table.index[:-1]:  # Exclude residual
            ss = anova_table.loc[term, 'sum_sq']
            df = anova_table.loc[term, 'df']
            omega_sq = (ss - df * ms_residual) / (ss_total + ms_residual)
            omega_squared[term] = max(0, omega_sq)
        
        # Calculate total variance explained
        r_squared = model.rsquared
        
        # Calculate variance explained by main effects and interactions separately
        main_effects_omega = sum(omega_squared[k] for k in omega_squared.keys() 
                                if ':' not in k)
        interaction_omega = sum(omega_squared[k] for k in omega_squared.keys() 
                               if ':' in k)
        
        print(f"R² = {r_squared:.4f}")
        print(f"Main effects ω² = {main_effects_omega:.4f}")
        print(f"Interaction ω² = {interaction_omega:.4f}")
        
        return omega_squared, r_squared, anova_table, main_effects_omega, interaction_omega
    
    except Exception as e:
        print(f"Error in ANOVA: {e}")
        import traceback
        traceback.print_exc()
        return {}, 0, None, 0, 0


def analyze_factors_with_interactions(averaged_compat):
    """Step 4b: Analyze factors including interaction effects"""
    print("\nStep 4b: Analyzing factors with interactions...")
    
    results = {}
    
    for pair in pairs_to_analyze:
        normalized_pair = normalize_pair(pair)
        pair_key = '-'.join(normalized_pair)
        
        print(f"\n{'='*60}")
        print(f"Processing: {pair[0]} vs {pair[1]}")
        print(f"{'='*60}")
        
        # Collect data
        data_records = []
        for (method, dataset, classifier), compat_dict in averaged_compat.items():
            if pair_key in compat_dict:
                data_records.append({
                    'method': method,
                    'dataset': dataset,
                    'model': classifier,
                    'compatibility': compat_dict[pair_key]
                })
        
        df = pd.DataFrame(data_records)
        print(f"Collected {len(df)} data points")
        
        # Compute omega-squared with interactions
        omega_sq, r_squared, anova_table, main_omega, inter_omega = \
            compute_omega_squared_with_interactions(df)
        
        results[pair] = {
            'omega_squared': omega_sq,
            'r_squared': r_squared,
            'anova_table': anova_table,
            'main_effects_omega': main_omega,
            'interaction_omega': inter_omega
        }
    
    # Visualize
    visualize_omega_squared_with_interactions(results)
    
    # Save detailed results
    save_interaction_results(results)
    
    return results


def visualize_omega_squared_with_interactions(results):
    """Visualize omega-squared including interactions"""
    
    n_pairs = len(results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(10*n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    
    for idx, (pair, result) in enumerate(results.items()):
        ax = axes[idx]
        omega_sq = result['omega_squared']
        r_squared = result['r_squared']
        
        if not omega_sq:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=14, fontweight='bold')
            continue
        
        # Sort by value
        sorted_items = sorted(omega_sq.items(), key=lambda x: x[1], reverse=True)
        
        # Clean up term names
        terms = []
        for item in sorted_items:
            term = item[0]
            term = term.replace('C(dataset)', 'Dataset')
            term = term.replace('C(model)', 'Model')
            term = term.replace('C(method)', 'Method')
            term = term.replace(':', ' × ')
            terms.append(term)
        
        values = [item[1] for item in sorted_items]
        
        # Color code: main effects (blue) vs interactions (orange)
        colors = ['#2E86AB' if '×' not in term else '#A23B72' for term in terms]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(terms))
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=17)
        ax.set_xlabel('Omega-squared (ω²)', fontsize=17)
        ax.set_title(f'{pair[0]} vs {pair[1]}\nR² = {r_squared:.3f}', 
                    fontsize=18, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.15 if values else 0.5)
        
        # Annotate bars
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                   f'{value:.3f}', ha='left', va='center', fontsize=16)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Main Effects'),
            Patch(facecolor='#A23B72', label='Interaction Effects')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=16)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(osp.join(results_dir, 'omega_squared_with_interactions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOmega-squared (with interactions) visualization saved to {results_dir}/omega_squared_with_interactions.png")


def save_interaction_results(results):
    """Save detailed interaction analysis results"""
    
    # Prepare data for JSON
    results_json = {}
    
    for pair, result in results.items():
        pair_key = f"{pair[0]}-{pair[1]}"
        
        omega_sq_dict = {}
        for term, value in result['omega_squared'].items():
            clean_term = term.replace('C(', '').replace(')', '')
            omega_sq_dict[clean_term] = float(value)
        
        results_json[pair_key] = {
            'r_squared': float(result['r_squared']),
            'main_effects_omega_squared': float(result['main_effects_omega']),
            'interaction_omega_squared': float(result['interaction_omega']),
            'omega_squared_by_term': omega_sq_dict
        }
        
        # Save ANOVA table if available
        if result['anova_table'] is not None:
            anova_df = result['anova_table'].copy()
            anova_csv_path = osp.join(results_dir, f'anova_table_{pair[0]}_vs_{pair[1]}.csv')
            anova_df.to_csv(anova_csv_path)
    
    # Save JSON
    json_path = osp.join(results_dir, 'interaction_analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Interaction analysis results saved to {json_path}")


def cramers_v(x, y):
    """Compute Cramér's V coefficient"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = len(x)
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def compute_dataset_features(dataset, seed):
    """Compute fairness difficulty features for a dataset"""
    # Load training data
    training_data_root = osp.join(
        script_dir, "../data/tabular", dataset,
        "prepared_data", f"seed_{seed}", "data_train.npy"
    )
    
    data = np.load(training_data_root)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Get sensitive attribute indices
    sensitive_indices = list(considered_sensitive_attributes[dataset].values())
    
    # Create intersectional sensitive attribute
    if len(sensitive_indices) > 1:
        # Combine all sensitive attributes
        sensitive_values = X[:, sensitive_indices].astype(int)
        # Create unique identifier for each combination
        multipliers = np.array([10**i for i in range(len(sensitive_indices))])
        intersectional_sensitive = np.dot(sensitive_values, multipliers)
    else:
        intersectional_sensitive = X[:, sensitive_indices[0]].astype(int)
    
    # Feature 1: Class Imbalance Ratio
    unique_labels, counts = np.unique(y, return_counts=True)
    class_imbalance = min(counts) / max(counts)
    
    # Feature 2: Average Correlation between Sensitive and Non-sensitive Attributes
    non_sensitive_indices = [i for i in range(X.shape[1]) if i not in sensitive_indices]
    cramers_v_values = []
    for idx in non_sensitive_indices:
        cv = cramers_v(intersectional_sensitive, X[:, idx].astype(int))
        cramers_v_values.append(abs(cv))
    avg_correlation = np.mean(cramers_v_values) if cramers_v_values else 0
    
    # Feature 3: Mutual Information between Sensitive Attribute and Label
    mi = normalized_mutual_info_score(intersectional_sensitive, y)
    
    return class_imbalance, avg_correlation, mi


def build_comprehensive_dataset(averaged_compat):
    """Step 5: Build comprehensive dataset with fairness difficulty features"""
    print("\nStep 5: Building comprehensive dataset...")
    
    records = []
    
    for method in METHODS:
        for dataset in DATASETS:
            for classifier in MODELS:
                key = (method, dataset, classifier)
                if key not in averaged_compat:
                    continue
                
                # Compute dataset features (average across seeds)
                features_per_seed = []
                for seed in RANDOM_SEED_LIST:
                    try:
                        features = compute_dataset_features(dataset, seed)
                        features_per_seed.append(features)
                    except:
                        continue
                
                if not features_per_seed:
                    continue
                
                avg_features = np.mean(features_per_seed, axis=0)
                class_imbalance, avg_correlation, mi = avg_features
                
                # Build record
                record = {
                    'method': method,
                    'dataset': dataset,
                    'model': classifier,
                    'class_imbalance': class_imbalance,
                    'avg_correlation': avg_correlation,
                    'mutual_information': mi
                }
                
                # Add compatibility for all metric pairs
                compat_dict = averaged_compat[key]
                for pair_key, value in compat_dict.items():
                    record[pair_key] = value
                
                records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save to CSV
    csv_path = osp.join(results_dir, 'comprehensive_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Comprehensive dataset saved to {csv_path}")
    print(f"Dataset shape: {df.shape}")
    
    return df


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_multiple_regressors(df, averaged_compat):
    """Step 6: Train multiple regressors with LODO cross-validation"""
    print("\nStep 6: Training multiple regressors (DT, RF, GB)...")
    
    # Define models
    models_to_train = {
        'Decision Tree': DecisionTreeRegressor(
            max_depth=3, 
            min_samples_leaf=5, 
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    all_results = {}
    all_trees = {}
    
    for pair in pairs_to_analyze:
        normalized_pair = normalize_pair(pair)
        pair_key = '-'.join(normalized_pair)
        
        if pair_key not in df.columns:
            print(f"Warning: {pair_key} not found in dataset")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {pair[0]} vs {pair[1]}")
        print(f"{'='*60}")
        
        # Prepare features and target
        X_categorical = df[['method', 'model']].values
        X_continuous = df[['class_imbalance', 'avg_correlation', 'mutual_information']].values
        y = df[pair_key].values
        dataset_labels = df['dataset'].values
        
        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical_encoded = encoder.fit_transform(X_categorical)
        
        # Combine features
        X = np.hstack([X_categorical_encoded, X_continuous])
        
        pair_results = {}
        
        # Train each model type
        for model_name, model_template in models_to_train.items():
            print(f"\n  Training {model_name}...")
            
            r2_scores = []
            mae_scores = []
            predictions_per_dataset = {}
            
            # Perform LODO CV
            for test_dataset in DATASETS:
                test_mask = dataset_labels == test_dataset
                train_mask = ~test_mask
                
                if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
                    continue
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                
                # Clone model for this fold
                from sklearn.base import clone
                model = clone(model_template)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Compute metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                
                predictions_per_dataset[test_dataset] = {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'r2': r2,
                    'mae': mae
                }
            
            # Compute statistics
            r2_mean = np.mean(r2_scores)
            mae_mean = np.mean(mae_scores)
            r2_std = np.std(r2_scores, ddof=1)
            mae_std = np.std(mae_scores, ddof=1)
            
            # CI using standard error
            r2_se = r2_std / np.sqrt(len(r2_scores))
            mae_se = mae_std / np.sqrt(len(mae_scores))
            
            # 95% CI
            from scipy.stats import t
            df_cv = len(r2_scores) - 1
            t_critical = t.ppf(0.975, df_cv) if df_cv > 0 else 1.96
            
            r2_ci = (r2_mean - t_critical * r2_se, r2_mean + t_critical * r2_se)
            mae_ci = (mae_mean - t_critical * mae_se, mae_mean + t_critical * mae_se)
            
            pair_results[model_name] = {
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'R2_CI': r2_ci,
                'MAE_mean': mae_mean,
                'MAE_std': mae_std,
                'MAE_CI': mae_ci,
                'predictions_per_dataset': predictions_per_dataset
            }
            
            print(f"    R² = {r2_mean:.3f} ± {r2_std:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
            print(f"    MAE = {mae_mean:.3f} ± {mae_std:.3f} [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
        
        all_results[pair] = pair_results
        
        # Train final models on full dataset for visualization (Decision Tree only)
        model_full = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)
        model_full.fit(X, y)
        all_trees[pair] = (model_full, encoder, X, y)
    
    # Save detailed results
    save_model_comparison_results(all_results)
    
    # Visualize comparison
    visualize_model_comparison(all_results)
    
    return all_results, all_trees


def save_model_comparison_results(all_results):
    """Save detailed model comparison results"""
    print("\nSaving model comparison results...")
    
    # Prepare data for JSON
    results_json = {}
    
    for pair, model_results in all_results.items():
        pair_key = f"{pair[0]}-{pair[1]}"
        results_json[pair_key] = {}
        
        for model_name, metrics in model_results.items():
            results_json[pair_key][model_name] = {
                'R2_mean': float(metrics['R2_mean']),
                'R2_std': float(metrics['R2_std']),
                'R2_CI_lower': float(metrics['R2_CI'][0]),
                'R2_CI_upper': float(metrics['R2_CI'][1]),
                'MAE_mean': float(metrics['MAE_mean']),
                'MAE_std': float(metrics['MAE_std']),
                'MAE_CI_lower': float(metrics['MAE_CI'][0]),
                'MAE_CI_upper': float(metrics['MAE_CI'][1])
            }
    
    json_path = osp.join(results_dir, 'model_comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Model comparison results saved to {json_path}")
    
    # Create summary table
    summary_data = []
    for pair, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            summary_data.append({
                'Metric Pair': f"{pair[0]} vs {pair[1]}",
                'Model': model_name,
                'R² Mean': metrics['R2_mean'],
                'R² Std': metrics['R2_std'],
                'R² CI Lower': metrics['R2_CI'][0],
                'R² CI Upper': metrics['R2_CI'][1],
                'MAE Mean': metrics['MAE_mean'],
                'MAE Std': metrics['MAE_std'],
                'MAE CI Lower': metrics['MAE_CI'][0],
                'MAE CI Upper': metrics['MAE_CI'][1]
            })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = osp.join(results_dir, 'model_comparison_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"Model comparison summary saved to {csv_path}")


def visualize_model_comparison(all_results):
    """Visualize comparison of different models"""
    print("\nVisualizing model comparison...")
    
    n_pairs = len(all_results)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 6*n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for idx, (pair, model_results) in enumerate(all_results.items()):
        # Plot R²
        ax_r2 = axes[idx, 0]
        
        r2_means = [model_results[m]['R2_mean'] for m in model_names]
        r2_stds = [model_results[m]['R2_std'] for m in model_names]
        
        x_pos = np.arange(len(model_names))
        bars = ax_r2.bar(x_pos, r2_means, yerr=r2_stds, 
                         color=colors, alpha=0.7, capsize=5)
        
        ax_r2.set_ylabel('R² Score', fontsize=14, fontweight='bold')
        ax_r2.set_title(f'{pair[0]} vs {pair[1]} - R² Comparison', 
                       fontsize=15, fontweight='bold')
        ax_r2.set_xticks(x_pos)
        ax_r2.set_xticklabels(model_names, fontsize=12)
        ax_r2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_r2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Annotate bars
        for bar, mean_val, std_val in zip(bars, r2_means, r2_stds):
            height = bar.get_height()
            y_pos = height + std_val + 0.02 if height >= 0 else height - std_val - 0.02
            ax_r2.text(bar.get_x() + bar.get_width()/2., y_pos,
                      f'{mean_val:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                      fontsize=11, fontweight='bold')
        
        # Plot MAE
        ax_mae = axes[idx, 1]
        
        mae_means = [model_results[m]['MAE_mean'] for m in model_names]
        mae_stds = [model_results[m]['MAE_std'] for m in model_names]
        
        bars = ax_mae.bar(x_pos, mae_means, yerr=mae_stds,
                         color=colors, alpha=0.7, capsize=5)
        
        ax_mae.set_ylabel('MAE', fontsize=14, fontweight='bold')
        ax_mae.set_title(f'{pair[0]} vs {pair[1]} - MAE Comparison', 
                        fontsize=15, fontweight='bold')
        ax_mae.set_xticks(x_pos)
        ax_mae.set_xticklabels(model_names, fontsize=12)
        ax_mae.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Annotate bars
        for bar, mean_val, std_val in zip(bars, mae_means, mae_stds):
            height = bar.get_height()
            ax_mae.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.005,
                       f'{mean_val:.3f}', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(osp.join(results_dir, 'model_comparison_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison visualization saved to {results_dir}/model_comparison_visualization.png")


def analyze_per_dataset_performance(all_results):
    """Analyze performance breakdown by dataset"""
    print("\nAnalyzing per-dataset performance...")
    
    for pair, model_results in all_results.items():
        print(f"\n{'='*60}")
        print(f"{pair[0]} vs {pair[1]}")
        print(f"{'='*60}")
        
        # Get all test datasets
        test_datasets = list(model_results['Decision Tree']['predictions_per_dataset'].keys())
        
        # Create comparison table
        comparison_data = []
        for dataset in test_datasets:
            row = {'Dataset': dataset}
            for model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                pred_data = model_results[model_name]['predictions_per_dataset'][dataset]
                row[f'{model_name}_R2'] = pred_data['r2']
                row[f'{model_name}_MAE'] = pred_data['mae']
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = osp.join(results_dir, f'per_dataset_performance_{pair[0]}_vs_{pair[1]}.csv')
        df_comparison.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\nPer-dataset performance:")
        print(df_comparison.to_string(index=False))
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        x_pos = np.arange(len(test_datasets))
        width = 0.25
        
        model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # R² comparison
        ax_r2 = axes[0]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            r2_values = [df_comparison.loc[df_comparison['Dataset'] == ds, f'{model_name}_R2'].values[0] 
                        for ds in test_datasets]
            ax_r2.bar(x_pos + i*width, r2_values, width, label=model_name, color=color, alpha=0.7)
        
        ax_r2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax_r2.set_title(f'R² by Dataset - {pair[0]} vs {pair[1]}', fontsize=14, fontweight='bold')
        ax_r2.set_xticks(x_pos + width)
        ax_r2.set_xticklabels(test_datasets, rotation=45, ha='right')
        ax_r2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_r2.legend(fontsize=10)
        ax_r2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # MAE comparison
        ax_mae = axes[1]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            mae_values = [df_comparison.loc[df_comparison['Dataset'] == ds, f'{model_name}_MAE'].values[0] 
                         for ds in test_datasets]
            ax_mae.bar(x_pos + i*width, mae_values, width, label=model_name, color=color, alpha=0.7)
        
        ax_mae.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax_mae.set_title(f'MAE by Dataset - {pair[0]} vs {pair[1]}', fontsize=14, fontweight='bold')
        ax_mae.set_xticks(x_pos + width)
        ax_mae.set_xticklabels(test_datasets, rotation=45, ha='right')
        ax_mae.legend(fontsize=10)
        ax_mae.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(osp.join(results_dir, f'per_dataset_comparison_{pair[0]}_vs_{pair[1]}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nPer-dataset analysis saved to {results_dir}")


def visualize_decision_trees(trees):
    """Visualize decision trees with annotations"""
    print("\nStep 6: Visualizing decision trees...")
    
    n_pairs = len(trees)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 10*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, (pair, (tree, encoder, X, y)) in enumerate(trees.items()):
        ax = axes[idx]
        
        # Get feature names
        method_features = [f"method_{m}" for m in encoder.categories_[0]]
        model_features = [f"model_{m}" for m in encoder.categories_[1]]
        continuous_features = ['class_imbalance', 'avg_correlation', 'mutual_information']
        feature_names = method_features + model_features + continuous_features
        
        # Plot tree
        plot_tree(tree, ax=ax, feature_names=feature_names, filled=True, 
                 rounded=True, fontsize=9, proportion=True)
        
        ax.set_title(f'{pair[0]} vs {pair[1]}', fontsize=16, fontweight='bold', pad=20)
        
        # Calculate leaf statistics
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        leaf_info = []
        for leaf_id in sorted(unique_leaves):
            mask = leaf_ids == leaf_id
            median_compat = np.median(y[mask])
            mean_compat = np.mean(y[mask])
            proportion = np.sum(mask) / len(y)
            n_samples = np.sum(mask)
            leaf_info.append(f"Leaf {leaf_id}: median={median_compat:.3f}, "
                           f"mean={mean_compat:.3f}, n={n_samples} ({proportion:.1%})")
        
        info_text = '\n'.join(leaf_info)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(osp.join(results_dir, 'decision_trees.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Decision trees visualization saved to {results_dir}/decision_trees.png")


def create_summary_comparison(main_effects_results, interaction_results):
    """Create a summary comparison of main effects vs. with interactions"""
    print("\nCreating summary comparison...")
    
    comparison_data = []
    
    for pair in pairs_to_analyze:
        # Main effects R²
        main_omega = (main_effects_results[pair]['dataset'] + 
                     main_effects_results[pair]['model'] + 
                     main_effects_results[pair]['method'])
        
        # With interactions R²
        inter_r2 = interaction_results[pair]['r_squared']
        inter_main = interaction_results[pair]['main_effects_omega']
        inter_interaction = interaction_results[pair]['interaction_omega']
        
        comparison_data.append({
            'Metric Pair': f"{pair[0]} vs {pair[1]}",
            'Main Effects ω² Sum': main_omega,
            'Full Model R²': inter_r2,
            'Main Effects ω²': inter_main,
            'Interactions ω²': inter_interaction,
            'Improvement': inter_r2 - main_omega
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = osp.join(results_dir, 'model_comparison_summary.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"Model comparison summary saved to {csv_path}")
    
    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_data))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_comparison['Main Effects ω² Sum'], width, 
                   label='Main Effects Only (ω² sum)', color='#1f77b4', alpha=0.7)
    bars2 = ax.bar(x, df_comparison['Main Effects ω²'], width, 
                   label='Main Effects (in full model)', color='#2ca02c', alpha=0.7)
    bars3 = ax.bar(x, df_comparison['Interactions ω²'], width, 
                   bottom=df_comparison['Main Effects ω²'],
                   label='Interaction Effects', color='#ff7f0e', alpha=0.7)
    
    ax.set_xlabel('Metric Pairs', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('Comparison: Main Effects vs. Full Model with Interactions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Metric Pair'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotate total R²
    for i, (idx, row) in enumerate(df_comparison.iterrows()):
        total_r2 = row['Full Model R²']
        ax.text(i, total_r2 + 0.02, f"R²={total_r2:.3f}", 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(osp.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison visualization saved to {results_dir}/model_comparison.png")
    
    return df_comparison


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("ML Fairness Tradeoffs Analysis - RQ5 (Complete with Interactions)")
    print("="*80)
    
    # Step 4: Compute averaged compatibility
    averaged_compat = compute_averaged_compatibility()
    print(f"Loaded compatibility data for {len(averaged_compat)} configurations\n")
    
    # Step 4: Analyze factors with omega-squared (main effects only)
    main_effects_results = analyze_factors_omega_squared(averaged_compat)
    
    # Step 4b: Analyze factors with interactions
    interaction_results = analyze_factors_with_interactions(averaged_compat)
    
    # Create comparison summary
    comparison_df = create_summary_comparison(main_effects_results, interaction_results)
    
    # Step 5: Build comprehensive dataset
    comprehensive_df = build_comprehensive_dataset(averaged_compat)
    print(f"Built dataset with {len(comprehensive_df)} samples\n")
    
    # Step 6: Train multiple regressors (DT, RF, GB)
    model_results, trees = train_multiple_regressors(comprehensive_df, averaged_compat)

    # Step 6b: Analyze per-dataset performance
    analyze_per_dataset_performance(model_results)

    # Step 6: Visualize decision trees
    visualize_decision_trees(trees)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"All results saved to: {results_dir}")
    print("\nGenerated files:")
    print("  1. omega_squared_main_effects.png - Main effects only")
    print("  2. omega_squared_with_interactions.png - Full model with interactions")
    print("  3. model_comparison.png - Comparison visualization")
    print("  4. model_comparison_summary.csv - Detailed comparison table")
    print("  5. interaction_analysis_results.json - Detailed ANOVA results")
    print("  6. anova_table_*.csv - Individual ANOVA tables")
    print("  7. comprehensive_dataset.csv - Full dataset")
    print("  8. model_comparison_results.json - Multi-model CV results (NEW)")
    print("  9. model_comparison_summary.csv - Multi-model summary table (NEW)")
    print(" 10. model_comparison_visualization.png - Multi-model comparison (NEW)")
    print(" 11. per_dataset_performance_*.csv - Per-dataset breakdown (NEW)")
    print(" 12. per_dataset_comparison_*.png - Per-dataset visualizations (NEW)")
    print(" 13. decision_trees.png - Decision tree visualizations")
    print("="*80)