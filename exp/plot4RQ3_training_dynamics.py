import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tools.config import RANDOM_SEED_LIST, NUM_CHECKPOINTS, METRICS, preprocessed_df_columns
from tools.utils import makedirs
from tools.models import compute_tabnet_checkpoint_importances

def load_compatibility_data():
    """
    Load all compatibility data across seeds and checkpoints
    Returns: dict[checkpoint_idx][metric_pair] = [values across seeds]
    """
    data = {idx: {} for idx in range(NUM_CHECKPOINTS)}
    
    for seed in RANDOM_SEED_LIST:
        for idx in range(NUM_CHECKPOINTS):
            compatibility_root = osp.join(
                script_dir, "../models", "census", "vanilla", 
                "tabnet", f"seed_{seed}", f"interactions_checkpoint_{idx}",
                "compatibility", "compatibility.json"
            )
            
            try:
                with open(compatibility_root, 'r') as f:
                    compatibility = json.load(f)
                    
                for pair, value in compatibility.items():
                    if pair not in data[idx]:
                        data[idx][pair] = []
                    data[idx][pair].append(value)
            except FileNotFoundError:
                print(f"Warning: File not found - {compatibility_root}")
                continue
    
    return data

def get_compatibility_value(avg_data, metric1, metric2):
    """
    Get compatibility value for a metric pair, handling order invariance
    Try both metric1-metric2 and metric2-metric1
    """
    key1 = f"{metric1}-{metric2}"
    key2 = f"{metric2}-{metric1}"
    
    if key1 in avg_data:
        return avg_data[key1]
    elif key2 in avg_data:
        return avg_data[key2]
    else:
        return None

def compute_average_compatibility(data):
    """
    Compute average compatibility for each metric pair at each checkpoint
    Returns: dict[metric_pair][checkpoint_idx] = average_value
    """
    avg_data = {}
    
    for idx in range(NUM_CHECKPOINTS):
        for pair, values in data[idx].items():
            if pair not in avg_data:
                avg_data[pair] = {}
            avg_data[pair][idx] = np.mean(values)
    
    return avg_data

def load_feature_importances():
    """
    Load and average feature importances across all seeds
    Returns: numpy array of shape (n_features, n_checkpoints)
    """
    all_importances = []
    
    for seed in RANDOM_SEED_LIST:
        feature_importances_root = osp.join(
            script_dir, "../models", "census", "vanilla", 
            "tabnet", f"seed_{seed}", "tabnet_feature_importances.npy"
        )
        
        try:
            importances = np.load(feature_importances_root)
            all_importances.append(importances)
        except FileNotFoundError:
            print(f"Warning: Feature importances not found for seed {seed}")
            continue
    
    if len(all_importances) == 0:
        return None
    
    # Average across all seeds
    avg_importances = np.mean(all_importances, axis=0)
    return avg_importances

def plot_combined_figure():
    """
    Plot combined figure with compatibility lower triangle and feature importance heatmap
    """
    print("Loading compatibility data...")
    raw_data = load_compatibility_data()
    avg_data = compute_average_compatibility(raw_data)
    
    print("Loading feature importances...")
    avg_importances = load_feature_importances()
    features_name = preprocessed_df_columns["census"][:-1]
    
    n_metrics = len(METRICS)
    
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(32, 14))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15, top=0.95)
    
    # Left subplot: Compatibility lower triangle
    gs_left = gs[0].subgridspec(n_metrics-1, n_metrics-1, hspace=0.3, wspace=0.3)
    
    for i in range(1, n_metrics):
        for j in range(i):
            metric_y = METRICS[i]
            metric_x = METRICS[j]
            
            pair_data = get_compatibility_value(avg_data, metric_x, metric_y)
            
            ax = fig.add_subplot(gs_left[i-1, j])
            
            if pair_data is not None:
                checkpoints = sorted(pair_data.keys())
                values = [pair_data[ckpt] for ckpt in checkpoints]
                
                colors = ['#2E86AB' if v >= 0 else '#A23B72' for v in values]
                
                bars = ax.bar(checkpoints, values, color=colors, width=0.8, 
                             edgecolor='black', linewidth=0.5)
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                
                ax.tick_params(axis='y', labelsize=12)
                ax.set_xticks([])
                
                y_max = max(abs(min(values)), abs(max(values)))
                ax.set_ylim(-y_max * 1.1, y_max * 1.1)
                
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            
            if j == 0:
                ax.set_ylabel(metric_y, fontsize=16, fontweight='bold')
            
            if i == n_metrics - 1:
                ax.set_xlabel(metric_x, fontsize=16, fontweight='bold')
    
    # Hide upper triangle
    for i in range(n_metrics-1):
        for j in range(i+1, n_metrics-1):
            ax = fig.add_subplot(gs_left[i, j])
            ax.axis('off')
    
    # Add title for left subplot (relative to left subplot position)
    left_bbox = gs[0].get_position(fig)
    fig.text(left_bbox.x0 + left_bbox.width / 2, 0.97, 
             'Inter-Metric Compatibility', ha='center', va='top', 
             fontsize=20, fontweight='bold')
    
    # Right subplot: Feature importance heatmap
    if avg_importances is not None:
        ax_right = fig.add_subplot(gs[1])
        
        # Use 'gray' colormap where higher values are brighter (white), lower values are darker (black)
        im = ax_right.imshow(avg_importances, cmap='gray', aspect='auto', 
                            interpolation='nearest')
        
        # Set ticks and labels
        ax_right.set_yticks(range(len(features_name)))
        ax_right.set_yticklabels(features_name, fontsize=16, fontweight='bold')
        
        ax_right.set_xticks(range(NUM_CHECKPOINTS))
        ax_right.set_xticklabels(range(NUM_CHECKPOINTS), fontsize=12)
        ax_right.set_xlabel('Checkpoint', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
        cbar.set_label('Importance', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Add title for right subplot (relative to right subplot position)
        right_bbox = gs[1].get_position(fig)
        fig.text(right_bbox.x0 + right_bbox.width / 2, 0.97, 
                 'Relative Feature Importance', ha='center', va='top', 
                 fontsize=20, fontweight='bold')
    
    results_dir = osp.join(script_dir, "results", "RQ3")
    makedirs(results_dir)
    
    output_path = osp.join(results_dir, "combined_compatibility_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    output_path_pdf = osp.join(results_dir, "combined_compatibility_importance.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF version saved to: {output_path_pdf}")
    
    plt.show()

if __name__ == "__main__":
    # Compute feature importances for all seeds if needed
    for seed in RANDOM_SEED_LIST:
        data_root = osp.join(
            script_dir, "../data/tabular", "census", "prepared_data", 
            f"seed_{seed}", "data_test_sampled.npy"
        )
        scaler_root = osp.join(
            script_dir, "../data/tabular", "census", "prepared_data", 
            f"seed_{seed}", "scaler.pkl"
        )
        checkpoint_base_path = osp.join(
            script_dir, "../models", "census", "vanilla", 
            "tabnet", f"seed_{seed}", "tabnet"
        )
        
        try:
            data_analyzed = np.load(data_root)
            scaler = joblib.load(scaler_root)
            test_X = scaler.transform(data_analyzed[:, :-1])
            compute_tabnet_checkpoint_importances(test_X, checkpoint_base_path)
        except Exception as e:
            print(f"Warning: Could not compute importances for seed {seed}: {e}")
            continue
    
    # Plot combined figure
    plot_combined_figure()