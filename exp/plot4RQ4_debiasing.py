import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import csv

from tools.config import (METHODS, DATASETS, MODELS, RANDOM_SEED_LIST, METRICS, 
                          COLORS_PAIR, METHODS_NAME, list_group_fairness, 
                          list_individual_fairness, list_utilities)
from tools.utils import makedirs


HIGH_COMPATIBILITY_THRESHOLD = 0.7
LOW_COMPATIBILITY_THRESHOLD = -0.07

def get_metric_category(metric):
    """Get the category of a metric"""
    if metric in list_group_fairness:
        return "Group Fairness"
    elif metric in list_individual_fairness:
        return "Individual Fairness"
    elif metric in list_utilities:
        return "Utility"
    else:
        return "Unknown"

def get_pair_category(metric1, metric2):
    """Get the category of a metric pair, handling order invariance"""
    cat1 = get_metric_category(metric1)
    cat2 = get_metric_category(metric2)
    
    # Normalize order to ensure invariance
    categories = sorted([cat1, cat2])
    pair_key = f"{categories[0]} vs {categories[1]}"
    
    return pair_key

def normalize_pair_key(metric1, metric2):
    """Normalize metric pair key to ensure order invariance"""
    return tuple(sorted([metric1, metric2]))

def parse_compatibility_key(key):
    """Parse key from compatibility.json"""
    parts = key.split('-')
    return parts[0], parts[1]


if __name__ == '__main__':
    # Collect compatibility data for all methods
    method_compatibilities = defaultdict(lambda: defaultdict(list))

    print("Reading compatibility data...")
    for method in METHODS:
        for dataset in DATASETS:
            for classifier in MODELS:
                for seed in RANDOM_SEED_LIST:
                    compatibility_path = osp.join(
                        script_dir, "../models", dataset, method, classifier, 
                        f"seed_{seed}", "interactions", "compatibility", "compatibility.json"
                    )
                    
                    if osp.exists(compatibility_path):
                        with open(compatibility_path, 'r') as f:
                            compatibility_data = json.load(f)
                        
                        # Process each metric pair
                        for key, value in compatibility_data.items():
                            metric1, metric2 = parse_compatibility_key(key)
                            normalized_key = normalize_pair_key(metric1, metric2)
                            method_compatibilities[method][normalized_key].append(value)
                    else:
                        print(f"Warning: File not found - {compatibility_path}")

    # Calculate average compatibility for each method
    method_avg_compatibilities = {}
    for method in METHODS:
        method_avg_compatibilities[method] = {}
        for pair_key, values in method_compatibilities[method].items():
            method_avg_compatibilities[method][pair_key] = np.mean(values)

    print(f"Processed {len(method_avg_compatibilities)} methods")

    # Save average compatibility data to JSON
    results_dir = osp.join(script_dir, "results", "RQ4")
    makedirs(results_dir)

    # Convert tuple keys to string for JSON serialization
    method_avg_compatibilities_serializable = {}
    for method, pairs in method_avg_compatibilities.items():
        method_avg_compatibilities_serializable[method] = {
            f"{pair[0]}-{pair[1]}": value for pair, value in pairs.items()
        }

    json_path = osp.join(results_dir, 'method_avg_compatibilities.json')
    with open(json_path, 'w') as f:
        json.dump(method_avg_compatibilities_serializable, f, indent=2)
    print(f"Average compatibilities saved to {json_path}")

    # Statistics for threshold-based pairs
    method_threshold_stats = {}

    for method in METHODS:
        avg_compat = method_avg_compatibilities[method]
        
        # Initialize statistics
        stats = {
            "high_compatibility": {
                "threshold": f">={HIGH_COMPATIBILITY_THRESHOLD}",
                "count": 0,
                "pairs": []
            },
            "low_compatibility": {
                "threshold": f"<={LOW_COMPATIBILITY_THRESHOLD}",
                "count": 0,
                "pairs": []
            }
        }
        
        # Collect pairs meeting thresholds
        for pair_key, compat_value in avg_compat.items():
            metric1, metric2 = pair_key
            pair_str = f"{metric1}-{metric2}"
            
            if compat_value >= HIGH_COMPATIBILITY_THRESHOLD:
                stats["high_compatibility"]["pairs"].append({
                    "pair": pair_str,
                    "compatibility": round(compat_value, 4)
                })
                stats["high_compatibility"]["count"] += 1
            
            if compat_value <= LOW_COMPATIBILITY_THRESHOLD:
                stats["low_compatibility"]["pairs"].append({
                    "pair": pair_str,
                    "compatibility": round(compat_value, 4)
                })
                stats["low_compatibility"]["count"] += 1
        
        # Sort pairs by compatibility value
        stats["high_compatibility"]["pairs"].sort(key=lambda x: x["compatibility"], reverse=True)
        stats["low_compatibility"]["pairs"].sort(key=lambda x: x["compatibility"])
        
        method_threshold_stats[method] = stats

    # Save threshold statistics to JSON
    threshold_stats_path = osp.join(results_dir, 'method_threshold_statistics.json')
    with open(threshold_stats_path, 'w') as f:
        json.dump(method_threshold_stats, f, indent=2)
    print(f"Threshold statistics saved to {threshold_stats_path}")

    # Calculate average compatibility by pair category for each method
    method_category_avg = {}

    for method in METHODS:
        avg_compat = method_avg_compatibilities[method]
        
        # Group by pair category
        category_values = defaultdict(list)
        all_values = []
        
        for pair_key, compat_value in avg_compat.items():
            metric1, metric2 = pair_key
            pair_category = get_pair_category(metric1, metric2)
            category_values[pair_category].append(compat_value)
            all_values.append(compat_value)
        
        # Calculate averages
        method_category_avg[method] = {}
        for category in COLORS_PAIR.keys():
            if category in category_values and len(category_values[category]) > 0:
                method_category_avg[method][category] = np.mean(category_values[category])
            else:
                method_category_avg[method][category] = np.nan
        
        # Calculate overall average
        method_category_avg[method]["All"] = np.mean(all_values) if all_values else np.nan

    # Save to CSV
    csv_path = osp.join(results_dir, 'method_category_compatibility.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        # Column headers: all pair categories plus "All"
        columns = list(COLORS_PAIR.keys()) + ["All"]
        
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Method"] + columns)
        
        # Write data for each method
        for method in METHODS:
            row = [method]
            for category in columns:
                value = method_category_avg[method][category]
                # Format to 4 decimal places, or empty if NaN
                if np.isnan(value):
                    row.append("")
                else:
                    row.append(f"{value:.4f}")
            writer.writerow(row)

    print(f"Category compatibility CSV saved to {csv_path}")

    # Create network graphs
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    # Use fixed layout (shared across all subplots)
    G_layout = nx.Graph()
    G_layout.add_nodes_from(METRICS)
    pos = nx.spring_layout(G_layout, seed=42, k=2, iterations=50)

    # Prepare legend elements
    from matplotlib.lines import Line2D
    legend_elements = []

    # Add color legend for pair categories
    for pair_type, color in COLORS_PAIR.items():
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=pair_type))

    # Add line style legend (using threshold variables)
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=3, 
                                linestyle='solid', label=f'Compatibility ≥ {HIGH_COMPATIBILITY_THRESHOLD}'))
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=3, 
                                linestyle='dashed', label=f'Compatibility ≤ {LOW_COMPATIBILITY_THRESHOLD}'))

    # Create subplots for each method
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        G = nx.Graph()
        G.add_nodes_from(METRICS)
        
        # Add edges
        edges_solid = []
        edges_dashed = []
        edge_colors_solid = []
        edge_colors_dashed = []
        
        avg_compat = method_avg_compatibilities[method]
        
        for pair_key, compat_value in avg_compat.items():
            metric1, metric2 = pair_key
            
            # Determine if edge should be drawn (using threshold variables)
            if compat_value >= HIGH_COMPATIBILITY_THRESHOLD:
                edges_solid.append((metric1, metric2))
                pair_category = get_pair_category(metric1, metric2)
                edge_colors_solid.append(COLORS_PAIR[pair_category])
            elif compat_value <= LOW_COMPATIBILITY_THRESHOLD:
                edges_dashed.append((metric1, metric2))
                pair_category = get_pair_category(metric1, metric2)
                edge_colors_dashed.append(COLORS_PAIR[pair_category])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', 
                            node_size=1200, ax=ax, alpha=0.9)
        
        # Draw solid edges
        if edges_solid:
            nx.draw_networkx_edges(G, pos, edgelist=edges_solid, 
                                edge_color=edge_colors_solid, 
                                width=3, style='solid', ax=ax, alpha=0.7)
        
        # Draw dashed edges
        if edges_dashed:
            nx.draw_networkx_edges(G, pos, edgelist=edges_dashed, 
                                edge_color=edge_colors_dashed, 
                                width=3, style='dashed', ax=ax, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
        
        # Set title
        ax.set_title(METHODS_NAME[method], fontsize=18, fontweight='bold')
        ax.axis('off')

    # Add shared legend at the top
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
            fontsize=16, frameon=True, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save figure
    fig_path = osp.join(results_dir, 'method_compatibility_networks.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()