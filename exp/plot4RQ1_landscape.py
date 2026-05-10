import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

from tools.utils import makedirs
from tools.config import DATASETS, MODELS, RANDOM_SEED_LIST, list_group_fairness, list_individual_fairness, list_utilities, COLORS_PAIR


def classify_metric_pair(metric_pair):
    """Classify metric pair into categories based on metric types"""
    metrics = metric_pair.split('-')
    if len(metrics) != 2:
        return None
    
    metric1, metric2 = metrics
    
    def get_metric_type(metric):
        if metric in list_group_fairness:
            return 'Group Fairness'
        elif metric in list_individual_fairness:
            return 'Individual Fairness'
        elif metric in list_utilities:
            return 'Utility'
        return None
    
    type1 = get_metric_type(metric1)
    type2 = get_metric_type(metric2)
    
    if type1 is None or type2 is None:
        return None
    
    if type1 == type2:
        return f'{type1} vs {type2}'
    else:
        types = sorted([type1, type2])
        return f'{types[0]} vs {types[1]}'


def read_compatibility_data():
    """Read all compatibility data from JSON files"""
    all_data = defaultdict(lambda: defaultdict(list))
    data_count = 0
    
    for dataset in DATASETS:
        for classifier in MODELS:
            for seed in RANDOM_SEED_LIST:
                compatibility_path = osp.join(
                    script_dir, "../models", dataset, "vanilla", 
                    classifier, f"seed_{seed}", "interactions",
                    "compatibility", "compatibility.json"
                )
                
                if osp.exists(compatibility_path):
                    with open(compatibility_path, 'r') as f:
                        data = json.load(f)
                        data_count += 1
                        
                    for metric_pair, value in data.items():
                        all_data[metric_pair][(dataset, classifier)].append(value)
    
    print(f"Successfully read {data_count} files")
    return all_data


def average_across_seeds(all_data):
    """Calculate average for each (dataset, model) combination across seeds"""
    averaged_data = {}
    for metric_pair, dataset_model_dict in all_data.items():
        averaged_data[metric_pair] = {
            key: np.mean(values) for key, values in dataset_model_dict.items()
        }
    return averaged_data


def prepare_boxplot_data(averaged_data):
    """Prepare data for boxplot with classification and colors"""
    boxplot_data = []
    metric_pairs = []
    colors = []
    categories = []
    
    for metric_pair, dataset_model_values in averaged_data.items():
        category = classify_metric_pair(metric_pair)
        if category is None:
            continue
        
        values = list(dataset_model_values.values())
        if len(values) > 0:
            boxplot_data.append(values)
            metric_pairs.append(metric_pair)
            categories.append(category)
            colors.append(COLORS_PAIR.get(category, '#999999'))
    
    return boxplot_data, metric_pairs, colors, categories


def sort_by_mean(boxplot_data, metric_pairs, colors, categories):
    """Sort all data by ascending mean compatibility"""
    means = [np.mean(data) for data in boxplot_data]
    sorted_indices = np.argsort(means)
    
    return (
        [boxplot_data[i] for i in sorted_indices],
        [metric_pairs[i] for i in sorted_indices],
        [colors[i] for i in sorted_indices],
        [categories[i] for i in sorted_indices]
    )


def create_boxplot(boxplot_data, metric_pairs, colors, categories):
    """Create and save the boxplot"""
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 17,
        'legend.fontsize': 17,
        'figure.figsize': (20, 10)
    })
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    positions = range(1, len(boxplot_data) + 1)
    bp = ax.boxplot(boxplot_data, 
                    positions=positions,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    
    ax.set_xlabel('Metric Pairs', fontsize=18, fontweight='bold')
    ax.set_ylabel('Compatibility', fontsize=18, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_pairs, rotation=90, ha='right')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='grey', linewidth=1, linestyle='-', alpha=0.8, zorder=1)
    ax.set_axisbelow(True)
    
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=label) 
                      for label, color in COLORS_PAIR.items()]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    results_dir = osp.join(script_dir, "results", "RQ1")
    makedirs(results_dir)
    output_path = osp.join(results_dir, 'compatibility_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBoxplot saved to: {output_path}")
    
    return categories


def print_statistics(metric_pairs, categories):
    """Print summary statistics"""
    print(f"\nTotal metric pairs: {len(metric_pairs)}")
    print(f"Total (dataset, model) combinations per metric pair: {len(DATASETS) * len(MODELS)}")
    print("\nCategory distribution:")
    
    category_counts = {}
    for category in categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")


if __name__ == '__main__':
    print("Reading data...")
    all_data = read_compatibility_data()
    
    print(f"\nTotal metric pairs found: {len(all_data)}")
    
    averaged_data = average_across_seeds(all_data)
    boxplot_data, metric_pairs, colors, categories = prepare_boxplot_data(averaged_data)
    
    print(f"Metric pairs to plot: {len(metric_pairs)}")
    print(f"Data points per metric pair: {len(boxplot_data[0]) if boxplot_data else 0}")
    
    boxplot_data, metric_pairs, colors, categories = sort_by_mean(
        boxplot_data, metric_pairs, colors, categories
    )
    
    categories = create_boxplot(boxplot_data, metric_pairs, colors, categories)
    print_statistics(metric_pairs, categories)
    
    plt.show()