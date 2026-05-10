"""
Step 3: Plot Per-Order Violin Plot for Marginal Sampling
=========================================================
Strictly replicate plot4RQ2_order.py style, using marginal sampling (K=100, no truncation)

Data: interactions_marginal/compatibility/{pair}_components.npy
- 6 datasets × 10 seeds × 36 metric pairs × 2^n components per pair

Usage:
    python 03_plot_marginal_per_order_violin.py

Output:
    figures/marginal/marginal_per_order_violin.png
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

from tools.utils import makedirs
from tools.config import (DATASETS, list_group_fairness, list_individual_fairness,
                          list_utilities, COLORS_PAIR, METRICS)


def classify_metric_pair(metric_pair):
    """Classify metric pair into categories based on metric types"""
    metric1, metric2 = metric_pair

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


def classify_order(order, n):
    """Classify interaction order into low/middle/high"""
    if order <= 0.3 * n:
        return 'Low'
    elif order < 0.7 * n:
        return 'Middle'
    else:
        return 'High'


def read_and_aggregate_data():
    """Read all data and aggregate by order category and metric pair category"""
    metric_pairs = list(combinations(METRICS, 2))

    # Structure: {order_category: {pair_category: [contributions]}}
    aggregated_data = {
        'Low': defaultdict(list),
        'Middle': defaultdict(list),
        'High': defaultdict(list)
    }

    total_files = 0
    missing_files = 0

    for dataset in DATASETS:
        for seed in range(10):
            # Read masks from marginal interactions
            masks_path = osp.join(
                script_dir, "../models", dataset, "vanilla",
                "lr", f"seed_{seed}", "interactions_marginal",
                "masks.npy"
            )

            if not osp.exists(masks_path):
                continue

            masks = np.load(masks_path)
            n = masks.shape[1]  # Number of features

            # Calculate interaction orders for all components
            orders = np.sum(masks, axis=1)

            # Process each metric pair
            for pair in metric_pairs:
                pair_category = classify_metric_pair(pair)
                if pair_category is None:
                    continue

                components_path = osp.join(
                    script_dir, "../models", dataset, "vanilla",
                    "lr", f"seed_{seed}", "interactions_marginal",
                    "compatibility", f"{pair[0]}-{pair[1]}_components.npy"
                )

                if not osp.exists(components_path):
                    missing_files += 1
                    continue

                components = np.load(components_path)
                total_files += 1

                # Aggregate components by order category
                for i, (order, contribution) in enumerate(zip(orders, components)):
                    order_category = classify_order(order, n)
                    aggregated_data[order_category][pair_category].append(contribution)

    print(f"Successfully read {total_files} component files")
    print(f"Missing files: {missing_files}")

    return aggregated_data


def create_grouped_violinplot(aggregated_data):
    """Create violin plot with grouped violins by order and colored by pair category"""
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 17,
        'figure.figsize': (16, 8)
    })

    fig, ax = plt.subplots(figsize=(16, 8))

    order_categories = ['Low', 'Middle', 'High']
    order_labels = {
        'Low': '[0, 0.3n]',
        'Middle': '(0.3n, 0.7n)',
        'High': '[0.7n, n]'
    }
    pair_categories = list(COLORS_PAIR.keys())

    # Prepare data for plotting
    all_data = []
    all_positions = []
    all_colors = []
    tick_positions = []
    tick_labels = []

    position = 1
    group_spacing = 2

    for order_idx, order_cat in enumerate(order_categories):
        group_start = position

        for pair_cat in pair_categories:
            data = aggregated_data[order_cat].get(pair_cat, [])

            if len(data) > 0:
                all_data.append(data)
                all_positions.append(position)
                all_colors.append(COLORS_PAIR[pair_cat])
                position += 1
            else:
                # Add empty data to maintain structure
                all_data.append([0])
                all_positions.append(position)
                all_colors.append(COLORS_PAIR[pair_cat])
                position += 1

        # Mark center of group for tick
        group_center = (group_start + position - 1) / 2
        tick_positions.append(group_center)
        tick_labels.append(order_labels[order_cat])

        position += group_spacing

    # Create violin plot WITHOUT extrema (no min/max markers)
    parts = ax.violinplot(all_data,
                          positions=all_positions,
                          widths=0.7,
                          showmeans=True,
                          showmedians=True,
                          showextrema=False)

    # Color violins
    for i, (pc, color) in enumerate(zip(parts['bodies'], all_colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Style the violin plot elements (now only cbars, cmedians, cmeans)
    for partname in ('cbars', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)

    # Make medians more visible
    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('red')
        parts['cmedians'].set_linewidth(2)

    # Make means more visible
    if 'cmeans' in parts:
        parts['cmeans'].set_edgecolor('blue')
        parts['cmeans'].set_linewidth(2)

    # Apply symlog scale to y-axis (non-linear)
    ax.set_yscale('symlog', linthresh=0.05, linscale=0.5)

    # Custom y-axis ticks for better readability
    yticks = [-1.0, -0.75, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.2f}' for y in yticks])

    # Set labels
    ax.set_ylabel('Contribution to Compatibility', fontsize=18, fontweight='bold')
    ax.set_xlabel('Interaction Order', fontsize=18, fontweight='bold')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='grey', linewidth=1.5, linestyle='-', alpha=0.8, zorder=1)
    ax.set_axisbelow(True)

    # Add vertical separators between groups
    for i in range(len(order_categories) - 1):
        separator_pos = tick_positions[i] + (tick_positions[i+1] - tick_positions[i]) / 2
        ax.axvline(x=separator_pos, color='black', linewidth=1.5, linestyle='--', alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=label, edgecolor='black')
                      for label, color in COLORS_PAIR.items()]
    # Add median and mean to legend
    legend_elements.extend([
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, label='Mean')
    ])
    ax.legend(handles=legend_elements,
             loc='lower left',
             bbox_to_anchor=(0, 1.02, 1, 0.2),
             ncol=3,
             mode="expand",
             borderaxespad=0,
             framealpha=0.9,
             frameon=True)

    plt.tight_layout()

    # Save figure
    output_dir = osp.join(script_dir, "figures", "marginal")
    makedirs(output_dir)
    output_path = osp.join(output_dir, 'marginal_per_order_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nViolin plot saved to: {output_path}")

    return aggregated_data


def print_statistics(aggregated_data):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)

    for order_cat in ['Low', 'Middle', 'High']:
        print(f"\n{order_cat} Order:")
        for pair_cat in COLORS_PAIR.keys():
            data = aggregated_data[order_cat].get(pair_cat, [])
            if len(data) > 0:
                print(f"  {pair_cat}: {len(data)} contributions")
                print(f"    Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")


def main():
    print("=" * 60)
    print("Step 3: Plot Per-Order Violin Plot for Marginal Sampling")
    print("K=100, No Truncation")
    print("=" * 60)

    print("\nReading and aggregating marginal sampling data...")
    aggregated_data = read_and_aggregate_data()

    print("\nCreating violin plot...")
    create_grouped_violinplot(aggregated_data)

    print_statistics(aggregated_data)

    print(f"\n{'='*60}")
    print("All steps complete!")


if __name__ == '__main__':
    main()