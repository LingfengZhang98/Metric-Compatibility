import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import json
import pandas as pd
from collections import defaultdict

from tools.utils import makedirs
from tools.config import DATASETS, MODELS, RANDOM_SEED_LIST, METHODS


if __name__ == '__main__':
    results_dir = osp.join(script_dir, "results")
    makedirs(results_dir)
    
    # Data structure to store metrics
    metrics_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    metrics_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    estimated_metrics_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    estimated_metrics_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Track metric order from first encountered JSON file
    fairness_order = []
    utilities_order = []
    order_recorded = False

    # Collect all metrics
    for method in METHODS:
        for dataset in DATASETS:
            for classifier in MODELS:
                for seed in RANDOM_SEED_LIST:
                    model_save_dir = osp.join(script_dir, "../models", dataset, method, classifier, f"seed_{seed}")
                    metrics_root = osp.join(model_save_dir, f"{classifier}_metrics.json")
                    estimated_metrics_root = osp.join(model_save_dir, f"{classifier}_estimated_metrics.json")
                    
                    # Read original metrics
                    if osp.exists(metrics_root):
                        with open(metrics_root, 'r') as f:
                            data = json.load(f)
                            
                            # Record order from first file
                            if not order_recorded:
                                if 'fairness' in data:
                                    fairness_order = list(data['fairness'].keys())
                                if 'utilities' in data:
                                    utilities_order = list(data['utilities'].keys())
                                order_recorded = True
                            
                            for category in ['utilities', 'fairness']:
                                if category in data:
                                    for metric_name, value in data[category].items():
                                        metrics_sum[method][category][metric_name] += value
                                        metrics_count[method][category][metric_name] += 1
                    
                    # Read estimated metrics
                    if osp.exists(estimated_metrics_root):
                        with open(estimated_metrics_root, 'r') as f:
                            data = json.load(f)
                            for category in ['utilities', 'fairness']:
                                if category in data:
                                    for metric_name, value in data[category].items():
                                        estimated_metrics_sum[method][category][metric_name] += value
                                        estimated_metrics_count[method][category][metric_name] += 1

    # Calculate averages
    metrics_avg = defaultdict(lambda: defaultdict(dict))
    estimated_metrics_avg = defaultdict(lambda: defaultdict(dict))

    for method in METHODS:
        for category in ['utilities', 'fairness']:
            for metric_name in metrics_sum[method][category]:
                if metrics_count[method][category][metric_name] > 0:
                    metrics_avg[method][category][metric_name] = \
                        metrics_sum[method][category][metric_name] / metrics_count[method][category][metric_name]
            
            for metric_name in estimated_metrics_sum[method][category]:
                if estimated_metrics_count[method][category][metric_name] > 0:
                    estimated_metrics_avg[method][category][metric_name] = \
                        estimated_metrics_sum[method][category][metric_name] / estimated_metrics_count[method][category][metric_name]

    # Build ordered metric list: fairness first, then utilities
    all_metrics = fairness_order + utilities_order

    # Build DataFrame
    rows = []
    for method in METHODS:
        # Original metrics row
        original_row = {'Method': method, 'Type': 'Original'}
        for metric_name in all_metrics:
            found = False
            for category in ['fairness', 'utilities']:
                if metric_name in metrics_avg[method][category]:
                    original_row[metric_name] = f"{metrics_avg[method][category][metric_name]:.4f}"
                    found = True
                    break
            if not found:
                original_row[metric_name] = "/"
        rows.append(original_row)
        
        # Estimated metrics row
        estimated_row = {'Method': method, 'Type': 'Approximate'}
        for metric_name in all_metrics:
            found = False
            for category in ['fairness', 'utilities']:
                if metric_name in estimated_metrics_avg[method][category]:
                    estimated_row[metric_name] = f"{estimated_metrics_avg[method][category][metric_name]:.4f}"
                    found = True
                    break
            if not found:
                estimated_row[metric_name] = "/"
        rows.append(estimated_row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    columns_order = ['Method', 'Type'] + all_metrics
    df = df[columns_order]

    # Save to CSV
    output_path = osp.join(results_dir, "metrics_comparison.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Results saved to: {output_path}")
    print(f"\nTotal combinations processed: {len(DATASETS)} datasets x {len(MODELS)} models x {len(RANDOM_SEED_LIST)} seeds")
    print(f"Methods: {len(METHODS)}")
    print(f"Metrics tracked: {len(all_metrics)}")
    print(f"Metric order: Fairness ({len(fairness_order)}) -> Utilities ({len(utilities_order)})")
    print("\nPreview:")
    print(df.to_string(index=False))