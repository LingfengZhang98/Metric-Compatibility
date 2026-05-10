import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import warnings
import json
from scipy.stats import kendalltau, spearmanr
import numpy as np
import pandas as pd

from tools.utils import calculate_mean_ci_for_correlations, makedirs, NumpyEncoder
from tools.config import DATASETS, MODELS, RANDOM_SEED_LIST, METRICS, METHODS


if __name__ == '__main__':
    results_dir = osp.join(script_dir, "results")
    makedirs(results_dir)
    
    # For case-level correlation
    lists_kendall_corr = {metric: [] for metric in METRICS}
    lists_spearman_corr = {metric: [] for metric in METRICS}
    
    # For method-level correlation: collect all values for each method
    method_actual_values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    method_estimated_values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    
    # For dataset-level correlation: collect values for each (dataset, method) pair
    # Structure: {metric: {(dataset, method): {'actual': [], 'estimated': []}}}
    dataset_method_values = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for dataset in DATASETS:
            for method in METHODS:
                dataset_method_values[metric][(dataset, method)] = {
                    'actual': [],
                    'estimated': []
                }
    
    # Add diagnostic statistics
    diagnostic_stats = {metric: {
        'total_comparisons': 0,
        'valid_comparisons': 0,
        'constant_x': 0,
        'constant_y': 0,
        'constant_both': 0,
        'insufficient_data': 0,
        'examples': []
    } for metric in METRICS}

    for dataset in DATASETS:
        for classifier in MODELS:
            for seed in RANDOM_SEED_LIST:
                metrics_by_method = {metric: [] for metric in METRICS}
                estimated_metrics_by_method = {metric: [] for metric in METRICS}
                
                # Collect data from all methods
                for method in METHODS:
                    model_save_dir = osp.join(script_dir, "../models", dataset, method, 
                                            classifier, f"seed_{seed}")
                    metrics_root = osp.join(model_save_dir, f"{classifier}_metrics.json")
                    estimated_metrics_root = osp.join(model_save_dir, f"{classifier}_estimated_metrics.json")
                
                    # Read metrics.json
                    if osp.exists(metrics_root):
                        with open(metrics_root, 'r') as f:
                            metrics_data = json.load(f)
                            
                        for metric in METRICS:
                            value = None
                            if metric in metrics_data.get("utilities", {}):
                                value = metrics_data["utilities"][metric]
                            elif metric in metrics_data.get("fairness", {}):
                                value = metrics_data["fairness"][metric]
                            
                            if value is not None:
                                metrics_by_method[metric].append(value)
                                # Store for method-level analysis
                                method_actual_values[method][metric].append(value)
                                # Store for dataset-level analysis
                                dataset_method_values[metric][(dataset, method)]['actual'].append(value)
                
                    # Read estimated_metrics.json
                    if osp.exists(estimated_metrics_root):
                        with open(estimated_metrics_root, 'r') as f:
                            estimated_data = json.load(f)
                            
                        for metric in METRICS:
                            value = None
                            if metric in estimated_data.get("utilities", {}):
                                value = estimated_data["utilities"][metric]
                            elif metric in estimated_data.get("fairness", {}):
                                value = estimated_data["fairness"][metric]
                            
                            if value is not None:
                                estimated_metrics_by_method[metric].append(value)
                                # Store for method-level analysis
                                method_estimated_values[method][metric].append(value)
                                # Store for dataset-level analysis
                                dataset_method_values[metric][(dataset, method)]['estimated'].append(value)
                
                # Calculate correlation coefficients and perform diagnostics (case-level)
                for metric in METRICS:
                    x = np.array(metrics_by_method[metric])
                    y = np.array(estimated_metrics_by_method[metric])
                    
                    diagnostic_stats[metric]['total_comparisons'] += 1
                    
                    # Detailed diagnostics
                    if len(x) < 2 or len(y) < 2:
                        diagnostic_stats[metric]['insufficient_data'] += 1
                        continue
                    
                    n_unique_x = len(np.unique(x))
                    n_unique_y = len(np.unique(y))
                    
                    # Record first few examples
                    if len(diagnostic_stats[metric]['examples']) < 3:
                        diagnostic_stats[metric]['examples'].append({
                            'dataset': dataset,
                            'classifier': classifier,
                            'seed': seed,
                            'x': x.tolist(),
                            'y': y.tolist(),
                            'n_unique_x': n_unique_x,
                            'n_unique_y': n_unique_y
                        })
                    
                    # Check for constants
                    is_constant_x = n_unique_x == 1
                    is_constant_y = n_unique_y == 1
                    
                    if is_constant_x and is_constant_y:
                        diagnostic_stats[metric]['constant_both'] += 1
                        continue
                    elif is_constant_x:
                        diagnostic_stats[metric]['constant_x'] += 1
                        continue
                    elif is_constant_y:
                        diagnostic_stats[metric]['constant_y'] += 1
                        continue
                    
                    # Data is valid, calculate correlation coefficients
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            tau, _ = kendalltau(x, y)
                            spearman, _ = spearmanr(x, y)
                        
                        if not np.isnan(tau) and not np.isnan(spearman):
                            lists_kendall_corr[metric].append(tau)
                            lists_spearman_corr[metric].append(spearman)
                            diagnostic_stats[metric]['valid_comparisons'] += 1
                    except Exception as e:
                        print(f"❌ Calculation error {dataset}-{classifier}-seed{seed}-{metric}: {e}")
    
    # ========================================
    # Calculate Case-level Results
    # ========================================
    
    print("\n" + "="*80)
    print("Case-level Correlation Results")
    print("="*80)
    
    case_level_results = {metric: {} for metric in METRICS}
    for metric in METRICS:
        if len(lists_kendall_corr[metric]) > 0:
            mean_and_ci_kendall = calculate_mean_ci_for_correlations(lists_kendall_corr[metric])
            mean_and_ci_spearman = calculate_mean_ci_for_correlations(lists_spearman_corr[metric])
            case_level_results[metric]["mean_Kendall"] = mean_and_ci_kendall["mean_correlation"]
            case_level_results[metric]["95% CI_Kendall"] = [mean_and_ci_kendall["ci_lower"], mean_and_ci_kendall["ci_upper"]]
            case_level_results[metric]["mean_Spearman"] = mean_and_ci_spearman["mean_correlation"]
            case_level_results[metric]["95% CI_Spearman"] = [mean_and_ci_spearman["ci_lower"], mean_and_ci_spearman["ci_upper"]]
            case_level_results[metric]["n_valid_comparisons"] = len(lists_kendall_corr[metric])
            
            print(f"\n【{metric}】")
            print(f"  Kendall τ: {mean_and_ci_kendall['mean_correlation']:.4f} "
                  f"[{mean_and_ci_kendall['ci_lower']:.4f}, {mean_and_ci_kendall['ci_upper']:.4f}]")
            print(f"  Spearman ρ: {mean_and_ci_spearman['mean_correlation']:.4f} "
                  f"[{mean_and_ci_spearman['ci_lower']:.4f}, {mean_and_ci_spearman['ci_upper']:.4f}]")
            print(f"  Valid comparisons: {len(lists_kendall_corr[metric])}")
        else:
            print(f"\n⚠️ {metric}: No valid correlation coefficient data")
            case_level_results[metric] = {
                "error": "No valid data",
                "reason": f"constant_x={diagnostic_stats[metric]['constant_x']}, "
                         f"constant_y={diagnostic_stats[metric]['constant_y']}, "
                         f"constant_both={diagnostic_stats[metric]['constant_both']}"
            }
    
    # ========================================
    # Calculate Dataset-level Results (NEW!)
    # ========================================
    
    print("\n" + "="*80)
    print("Dataset-level Correlation Results (Recommended)")
    print("="*80)
    
    dataset_level_results = {metric: {} for metric in METRICS}
    
    for metric in METRICS:
        # Aggregate: average over (7 models × 10 seeds) for each (dataset, method)
        aggregated_actual = []
        aggregated_estimated = []
        data_points_info = []
        
        for dataset in DATASETS:
            for method in METHODS:
                key = (dataset, method)
                actual_vals = dataset_method_values[metric][key]['actual']
                estimated_vals = dataset_method_values[metric][key]['estimated']
                
                # Only include if we have data
                if len(actual_vals) > 0 and len(estimated_vals) > 0:
                    mean_actual = np.mean(actual_vals)
                    mean_estimated = np.mean(estimated_vals)
                    
                    aggregated_actual.append(mean_actual)
                    aggregated_estimated.append(mean_estimated)
                    data_points_info.append({
                        'dataset': dataset,
                        'method': method,
                        'n_observations': len(actual_vals),
                        'actual_mean': mean_actual,
                        'estimated_mean': mean_estimated
                    })
        
        # Calculate correlation on aggregated data
        if len(aggregated_actual) >= 10:  # At least 10 points for reliable correlation
            x = np.array(aggregated_actual)
            y = np.array(aggregated_estimated)
            
            n_unique_x = len(np.unique(x))
            n_unique_y = len(np.unique(y))
            
            if n_unique_x > 1 and n_unique_y > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        tau, p_tau = kendalltau(x, y)
                        rho, p_rho = spearmanr(x, y)
                    
                    if not np.isnan(tau) and not np.isnan(rho):
                        dataset_level_results[metric] = {
                            'Kendall_tau': tau,
                            'Kendall_p_value': p_tau,
                            'Spearman_rho': rho,
                            'Spearman_p_value': p_rho,
                            'n_points': len(aggregated_actual),
                            'n_datasets': len(DATASETS),
                            'n_methods': len(METHODS),
                            'data_points': data_points_info
                        }
                        
                        print(f"\n【{metric}】")
                        print(f"  Kendall τ: {tau:.4f} (p={p_tau:.4f})")
                        print(f"  Spearman ρ: {rho:.4f} (p={p_rho:.4f})")
                        print(f"  Sample size: n={len(aggregated_actual)} ({len(DATASETS)} datasets × {len(METHODS)} methods)")
                        print(f"  Each point averages {len(MODELS)} models × {len(RANDOM_SEED_LIST)} seeds = {len(MODELS) * len(RANDOM_SEED_LIST)} observations")
                    else:
                        dataset_level_results[metric] = {"error": "NaN in correlation"}
                        print(f"\n⚠️ {metric}: NaN in correlation")
                except Exception as e:
                    dataset_level_results[metric] = {"error": f"Calculation failed: {str(e)}"}
                    print(f"\n❌ {metric}: Calculation failed - {e}")
            else:
                dataset_level_results[metric] = {
                    "error": "Constant values",
                    "n_unique_actual": n_unique_x,
                    "n_unique_estimated": n_unique_y
                }
                print(f"\n⚠️ {metric}: Constant values (unique actual={n_unique_x}, unique estimated={n_unique_y})")
        else:
            dataset_level_results[metric] = {
                "error": "Insufficient data points",
                "n_points": len(aggregated_actual)
            }
            print(f"\n⚠️ {metric}: Insufficient data points (n={len(aggregated_actual)} < 10)")
    
    # ========================================
    # Calculate Method-level Results
    # ========================================
    
    print("\n" + "="*80)
    print("Method-level Correlation Results")
    print("="*80)
    
    method_level_results = {metric: {} for metric in METRICS}
    
    for metric in METRICS:
        method_means_actual = []
        method_means_estimated = []
        method_names = []
        
        # Calculate mean for each method
        for method in METHODS:
            if len(method_actual_values[method][metric]) > 0 and len(method_estimated_values[method][metric]) > 0:
                mean_actual = np.mean(method_actual_values[method][metric])
                mean_estimated = np.mean(method_estimated_values[method][metric])
                
                method_means_actual.append(mean_actual)
                method_means_estimated.append(mean_estimated)
                method_names.append(method)
        
        # Calculate correlation if we have enough methods
        if len(method_means_actual) >= 2:
            n_unique_actual = len(np.unique(method_means_actual))
            n_unique_estimated = len(np.unique(method_means_estimated))
            
            if n_unique_actual > 1 and n_unique_estimated > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        tau_method, p_tau = kendalltau(method_means_actual, method_means_estimated)
                        rho_method, p_rho = spearmanr(method_means_actual, method_means_estimated)
                    
                    if not np.isnan(tau_method) and not np.isnan(rho_method):
                        method_level_results[metric] = {
                            'Kendall_tau': tau_method,
                            'Kendall_p_value': p_tau,
                            'Spearman_rho': rho_method,
                            'Spearman_p_value': p_rho,
                            'n_methods': len(method_names),
                            'method_names': method_names,
                            'actual_means': method_means_actual,
                            'estimated_means': method_means_estimated
                        }
                        
                        print(f"\n【{metric}】")
                        print(f"  Kendall τ: {tau_method:.4f} (p={p_tau:.4f})")
                        print(f"  Spearman ρ: {rho_method:.4f} (p={p_rho:.4f})")
                        print(f"  Number of methods: {len(method_names)}")
                    else:
                        method_level_results[metric] = {"error": "NaN in correlation"}
                        print(f"\n⚠️ {metric}: NaN in correlation")
                except Exception as e:
                    method_level_results[metric] = {"error": f"Calculation failed: {str(e)}"}
                    print(f"\n❌ {metric}: Calculation failed - {e}")
            else:
                method_level_results[metric] = {
                    "error": "Constant values",
                    "n_unique_actual": n_unique_actual,
                    "n_unique_estimated": n_unique_estimated
                }
                print(f"\n⚠️ {metric}: Constant values (unique actual={n_unique_actual}, unique estimated={n_unique_estimated})")
        else:
            method_level_results[metric] = {"error": "Insufficient methods", "n_methods": len(method_means_actual)}
            print(f"\n⚠️ {metric}: Insufficient methods ({len(method_means_actual)} < 2)")
    
    # ========================================
    # Print Diagnostic Report
    # ========================================
    
    print("\n" + "="*80)
    print("Diagnostic Report: Data Quality Analysis")
    print("="*80)
    
    for metric in METRICS:
        stats = diagnostic_stats[metric]
        total = stats['total_comparisons']
        valid = stats['valid_comparisons']
        
        print(f"\n【{metric}】")
        print(f"  Total comparisons: {total}")
        print(f"  Valid comparisons: {valid} ({valid/total*100:.1f}%)")
        print(f"  Insufficient data: {stats['insufficient_data']}")
        print(f"  X is constant: {stats['constant_x']}")
        print(f"  Y is constant: {stats['constant_y']}")
        print(f"  Both X and Y are constant: {stats['constant_both']}")
        
        # Display examples
        if stats['examples']:
            print(f"  Example data (first 3):")
            for i, ex in enumerate(stats['examples'][:3], 1):
                print(f"    Example {i}: {ex['dataset']}-{ex['classifier']}-seed{ex['seed']}")
                print(f"        Actual values: {ex['x']}")
                print(f"        Estimated values: {ex['y']}")
                print(f"        Unique values: X={ex['n_unique_x']}, Y={ex['n_unique_y']}")
    
    print("\n" + "="*80)
    
    # ========================================
    # Create CSV Summary Table (NEW!)
    # ========================================
    
    print("\n" + "="*80)
    print("Creating CSV Summary Table")
    print("="*80)
    
    # Create multi-index for rows: (Level, Correlation Type)
    row_index = pd.MultiIndex.from_tuples([
        ('Case-level', 'Kendall τ'),
        ('Case-level', 'Spearman ρ'),
        ('Dataset-level', 'Kendall τ'),
        ('Dataset-level', 'Spearman ρ'),
        ('Method-level', 'Kendall τ'),
        ('Method-level', 'Spearman ρ'),
    ], names=['Level', 'Correlation'])
    
    # Initialize DataFrame
    df_summary = pd.DataFrame(index=row_index, columns=METRICS)
    
    # Fill in the values
    for metric in METRICS:
        # Case-level
        if 'mean_Kendall' in case_level_results[metric]:
            val = case_level_results[metric]['mean_Kendall']
            df_summary.loc[('Case-level', 'Kendall τ'), metric] = f"{val:.4f}"
        else:
            df_summary.loc[('Case-level', 'Kendall τ'), metric] = "N/A"
        
        if 'mean_Spearman' in case_level_results[metric]:
            val = case_level_results[metric]['mean_Spearman']
            df_summary.loc[('Case-level', 'Spearman ρ'), metric] = f"{val:.4f}"
        else:
            df_summary.loc[('Case-level', 'Spearman ρ'), metric] = "N/A"
        
        # Dataset-level (with significance stars)
        if 'Kendall_tau' in dataset_level_results[metric]:
            val = dataset_level_results[metric]['Kendall_tau']
            p_val = dataset_level_results[metric]['Kendall_p_value']
            star = "*" if p_val < 0.05 else ""
            df_summary.loc[('Dataset-level', 'Kendall τ'), metric] = f"{val:.4f}{star}"
        else:
            df_summary.loc[('Dataset-level', 'Kendall τ'), metric] = "N/A"
        
        if 'Spearman_rho' in dataset_level_results[metric]:
            val = dataset_level_results[metric]['Spearman_rho']
            p_val = dataset_level_results[metric]['Spearman_p_value']
            star = "*" if p_val < 0.05 else ""
            df_summary.loc[('Dataset-level', 'Spearman ρ'), metric] = f"{val:.4f}{star}"
        else:
            df_summary.loc[('Dataset-level', 'Spearman ρ'), metric] = "N/A"
        
        # Method-level (with significance stars)
        if 'Kendall_tau' in method_level_results[metric]:
            val = method_level_results[metric]['Kendall_tau']
            p_val = method_level_results[metric]['Kendall_p_value']
            star = "*" if p_val < 0.05 else ""
            df_summary.loc[('Method-level', 'Kendall τ'), metric] = f"{val:.4f}{star}"
        else:
            df_summary.loc[('Method-level', 'Kendall τ'), metric] = "N/A"
        
        if 'Spearman_rho' in method_level_results[metric]:
            val = method_level_results[metric]['Spearman_rho']
            p_val = method_level_results[metric]['Spearman_p_value']
            star = "*" if p_val < 0.05 else ""
            df_summary.loc[('Method-level', 'Spearman ρ'), metric] = f"{val:.4f}{star}"
        else:
            df_summary.loc[('Method-level', 'Spearman ρ'), metric] = "N/A"
    
    # Save to CSV
    csv_path = osp.join(results_dir, "correlation_summary.csv")
    df_summary.to_csv(csv_path)
    
    print(f"✓ CSV summary saved to: {csv_path}")
    print(f"\nPreview of CSV table:")
    print(df_summary)
    
    # ========================================
    # Save Results
    # ========================================
    
    combined_results = {
        "case_level": case_level_results,
        "dataset_level": dataset_level_results,
        "method_level": method_level_results
    }
    
    # Save combined results
    with open(osp.join(results_dir, "consistency_corr.json"), "w") as f:
        json.dump(combined_results, f, indent=2, cls=NumpyEncoder)
    
    # Save diagnostic report
    with open(osp.join(results_dir, "diagnostic_report.json"), "w") as f:
        json.dump(diagnostic_stats, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Combined results saved to: {osp.join(results_dir, 'consistency_corr.json')}")
    print(f"✓ Diagnostic report saved to: {osp.join(results_dir, 'diagnostic_report.json')}")
    print(f"\n📊 Summary:")
    print(f"  - Case-level: Averaged correlations across {len(DATASETS)} × {len(MODELS)} × {len(RANDOM_SEED_LIST)} = {len(DATASETS) * len(MODELS) * len(RANDOM_SEED_LIST)} cases")
    print(f"  - Dataset-level (RECOMMENDED): Correlation on {len(DATASETS)} × {len(METHODS)} = {len(DATASETS) * len(METHODS)} aggregated points")
    print(f"  - Method-level: Correlation of {len(METHODS)} method averages")