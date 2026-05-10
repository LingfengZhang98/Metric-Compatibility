import os
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools.utils import makedirs
from tools.config import DATASETS, MODELS, RANDOM_SEED_LIST, METRICS, COLORS_METRIC, MARKERS_MODEL, DATASETS_NAME, MODELS_NAME


if __name__ == '__main__':
    # Create figure with 2 rows and 3 columns, aspect ratio 2:1
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    
    # Process each dataset separately
    for dataset_idx, dataset in enumerate(DATASETS):
        print(f"Processing dataset: {dataset}")
        ax = axes[dataset_idx]
        
        # Store all averaged vectors and their corresponding labels for this dataset
        all_vectors = []
        all_labels = []  # Store (model_name, metric) tuples
        
        # Collect data: average across random seeds for each (model, metric) combination
        for classifier in MODELS:
            for metric in METRICS:
                vectors_for_seeds = []
                
                for seed in RANDOM_SEED_LIST:
                    contribution_path = osp.join(
                        script_dir, "../models", dataset, "vanilla", 
                        classifier, f"seed_{seed}", "interactions",
                        "contribution_vectors", f"{metric}_interaction_contribution.npy"
                    )
                    
                    # Check if file exists
                    if osp.exists(contribution_path):
                        vector = np.load(contribution_path)
                        vectors_for_seeds.append(vector)
                    else:
                        print(f"Warning: File not found - {contribution_path}")
                
                # If data exists for this combination, compute the mean
                if len(vectors_for_seeds) > 0:
                    mean_vector = np.mean(vectors_for_seeds, axis=0)
                    all_vectors.append(mean_vector)
                    all_labels.append((classifier, metric))
                else:
                    print(f"Warning: No data found for {dataset}-{classifier}-{metric}")
        
        # Skip if no data found for this dataset
        if len(all_vectors) == 0:
            print(f"No data found for dataset {dataset}, skipping...")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Convert to numpy array
        X = np.array(all_vectors)
        print(f"Data shape for {dataset}: {X.shape}")
        
        # Perform PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Print explained variance ratio
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Plot each data point
        for i, (classifier, metric) in enumerate(all_labels):
            color = COLORS_METRIC[metric]
            marker = MARKERS_MODEL.get(classifier, 'o')
            
            ax.scatter(X_pca[i, 0], X_pca[i, 1], 
                      c=color, marker=marker, s=150, 
                      alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Set labels with larger font
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                     fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', 
                     fontsize=14)
        
        # Set title with dataset full name
        ax.set_title(DATASETS_NAME.get(dataset, dataset), 
                    fontsize=16, fontweight='bold')
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create shared legends
    from matplotlib.lines import Line2D
    
    # Metric legend (colors)
    metric_legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=COLORS_METRIC[metric], 
               markersize=14, label=metric, markeredgecolor='black')
        for metric in METRICS
    ]
    
    # Model legend (shapes) - using formal names
    model_legend_elements = [
        Line2D([0], [0], marker=MARKERS_MODEL.get(classifier, 'o'), color='w', 
               markerfacecolor='gray', markersize=14, 
               label=MODELS_NAME.get(classifier, classifier), markeredgecolor='black')
        for classifier in MODELS
    ]
    
    # Add shared legends to the figure with tighter positioning
    legend1 = fig.legend(handles=metric_legend_elements, 
                        title='Metrics', loc='center left', 
                        bbox_to_anchor=(0.98, 0.7), frameon=True,
                        fontsize=14, title_fontsize=16)
    
    legend2 = fig.legend(handles=model_legend_elements, 
                        title='Models', loc='center left', 
                        bbox_to_anchor=(0.98, 0.3), frameon=True,
                        fontsize=14, title_fontsize=16)
    
    # Adjust layout with more space for legends
    plt.tight_layout(rect=[0, 0, 0.96, 1])
    
    # Save figure
    results_dir = osp.join(script_dir, "results", "RQ2")
    makedirs(results_dir)
    output_path = osp.join(results_dir, "all_datasets_pca_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Display figure
    plt.show()
    plt.close()
    
    print("All visualizations completed!")