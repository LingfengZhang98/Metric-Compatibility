"""
Plot Compatibility Landscape for Median and Mode Baselines
Baseline Robustness Analysis - Step 4
Vertical layout with independently sorted metric pairs.
Extended to 6 datasets: census, ufrgs, compas, diabetes, bank, heart
"""

import os
import os.path as osp
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")
sys.path.append(project_root)

from tools.utils import makedirs
from tools.config import list_group_fairness, list_individual_fairness, list_utilities

COLORS_PAIR = {
    'Group Fairness vs Utility': '#4477AA',
    'Individual Fairness vs Utility': '#EE6677',
    'Group Fairness vs Individual Fairness': '#228833',
    'Group Fairness vs Group Fairness': '#CCBB44',
    'Individual Fairness vs Individual Fairness': '#66CCEE',
    'Utility vs Utility': '#AA3377'
}

DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
METHOD = "vanilla"
CLASSIFIER = "lr"
SEEDS = list(range(10))

BASELINES = {
    "Median": "interactions_median",
    "Mode": "interactions_mode"
}


def classify_metric_pair(metric_pair):
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
        return f'{type1} vs {type1}'
    else:
        types = sorted([type1, type2])
        return f'{types[0]} vs {types[1]}'


def read_compatibility_data_for_all_datasets(baseline_suffix):
    """Read compatibility data from all datasets and combine."""
    all_data = defaultdict(list)
    
    for dataset in DATASETS:
        for seed in SEEDS:
            compatibility_path = osp.join(
                project_root, "models", dataset, METHOD, CLASSIFIER, 
                f"seed_{seed}", baseline_suffix,
                "compatibility", "compatibility.json"
            )
            if osp.exists(compatibility_path):
                with open(compatibility_path, 'r') as f:
                    data = json.load(f)
                for metric_pair, value in data.items():
                    all_data[metric_pair].append(value)
    
    return all_data


def sort_by_mean(all_data):
    metric_means = {}
    for metric_pair, values in all_data.items():
        if len(values) > 0:
            metric_means[metric_pair] = np.mean(values)
    return sorted(metric_means.keys(), key=lambda x: metric_means[x])


def create_vertical_comparison_plot():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (20, 14)
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    
    all_data_median = read_compatibility_data_for_all_datasets(BASELINES["Median"])
    all_data_mode = read_compatibility_data_for_all_datasets(BASELINES["Mode"])
    
    sorted_pairs_median = sort_by_mean(all_data_median)
    sorted_pairs_mode = sort_by_mean(all_data_mode)
    
    # Plot Median (top panel)
    ax_median = axes[0]
    median_colors = [COLORS_PAIR.get(classify_metric_pair(mp), '#999999') for mp in sorted_pairs_median]
    median_boxplot_data = [all_data_median[mp] for mp in sorted_pairs_median]
    
    positions_median = range(1, len(sorted_pairs_median) + 1)
    bp1 = ax_median.boxplot(median_boxplot_data,
                            positions=positions_median,
                            patch_artist=True,
                            showfliers=True,
                            flierprops=dict(marker='o', markersize=3, alpha=0.5),
                            widths=0.65)
    
    for patch, color in zip(bp1['boxes'], median_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for flier, color in zip(bp1['fliers'], median_colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    
    ax_median.set_ylabel('Compatibility', fontsize=14, fontweight='bold')
    ax_median.set_title('Median Baseline', fontsize=15, fontweight='bold', pad=10)
    ax_median.set_xticks(positions_median)
    ax_median.set_xticklabels(sorted_pairs_median, rotation=90, ha='right', fontsize=9)
    ax_median.grid(axis='y', alpha=0.3, linestyle='--')
    ax_median.axhline(y=0, color='grey', linewidth=1, linestyle='-', alpha=0.8, zorder=1)
    ax_median.set_axisbelow(True)
    
    # Plot Mode (bottom panel)
    ax_mode = axes[1]
    mode_colors = [COLORS_PAIR.get(classify_metric_pair(mp), '#999999') for mp in sorted_pairs_mode]
    mode_boxplot_data = [all_data_mode[mp] for mp in sorted_pairs_mode]
    
    positions_mode = range(1, len(sorted_pairs_mode) + 1)
    bp2 = ax_mode.boxplot(mode_boxplot_data,
                          positions=positions_mode,
                          patch_artist=True,
                          showfliers=True,
                          flierprops=dict(marker='o', markersize=3, alpha=0.5),
                          widths=0.65)
    
    for patch, color in zip(bp2['boxes'], mode_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for flier, color in zip(bp2['fliers'], mode_colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    
    ax_mode.set_xlabel('Metric Pairs (sorted by mean compatibility)', fontsize=14, fontweight='bold')
    ax_mode.set_ylabel('Compatibility', fontsize=14, fontweight='bold')
    ax_mode.set_title('Mode Baseline', fontsize=15, fontweight='bold', pad=10)
    ax_mode.set_xticks(positions_mode)
    ax_mode.set_xticklabels(sorted_pairs_mode, rotation=90, ha='right', fontsize=9)
    ax_mode.grid(axis='y', alpha=0.3, linestyle='--')
    ax_mode.axhline(y=0, color='grey', linewidth=1, linestyle='-', alpha=0.8, zorder=1)
    ax_mode.set_axisbelow(True)
    
    # Add legend between title and first subplot
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=label) 
                      for label, color in COLORS_PAIR.items()]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=len(COLORS_PAIR), framealpha=0.9,
              bbox_to_anchor=(0.5, 0.95))
    
    dataset_names = {"census": "Census", "ufrgs": "UFRGS", "compas": "COMPAS", 
                    "diabetes": "Diabetes", "bank": "Bank", "heart": "Heart"}
    dataset_str = ", ".join([dataset_names[d] for d in DATASETS])
    
    fig.suptitle(f'Compatibility Landscape: Median vs Mode Baseline\n({dataset_str}, Vanilla, LR, 6 Datasets × 10 Seeds)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)
    
    output_dir = osp.join(script_dir, "figures")
    makedirs(output_dir)
    output_path = osp.join(output_dir, 'landscape_median_vs_mode.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    plt.close()


def main():
    print("=" * 60)
    print("Compatibility Landscape: Median vs Mode Baseline")
    print("=" * 60)
    print(f"Datasets: {DATASETS}")
    print(f"Method: {METHOD}, Classifier: {CLASSIFIER}")
    print(f"Seeds: {SEEDS} (10 per dataset, 60 total combinations)")
    print(f"Baselines: {list(BASELINES.keys())}")
    print("=" * 60)
    
    create_vertical_comparison_plot()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
