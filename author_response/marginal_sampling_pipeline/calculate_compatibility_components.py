"""
Calculate marginal sampling compatibility components and save as _components.npy
This replicates calculate_compatibility_for_median_mode.py logic for marginal sampling
"""

import warnings
warnings.filterwarnings('ignore')

import os
import os.path as osp
import sys
import json
from itertools import combinations

import numpy as np

script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")
sys.path.append(project_root)

from tools.config import METRICS
from tools.utils import makedirs

DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
METHOD = "vanilla"
CLASSIFIER = "lr"
SEEDS = list(range(10))


def cosine_similarity_components(a, b):
    """
    Calculate cosine similarity and component vector.
    component_vector[i] = (a[i] * b[i]) / (||a|| * ||b||)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0, np.zeros_like(a)

    component_vector = (a * b) / (norm_a * norm_b)
    similarity = np.sum(component_vector)

    return similarity, component_vector


def calculate_marginal_components():
    """
    Calculate marginal sampling compatibility components for all metric pairs.
    Saves _components.npy files similar to the mean baseline structure.
    """
    print("=" * 60)
    print("Calculating Marginal Sampling Compatibility Components")
    print("=" * 60)

    metric_pairs = list(combinations(METRICS, 2))
    print(f"Total metric pairs: {len(metric_pairs)}")

    total_pairs = 0
    total_files = 0

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")

        for seed in SEEDS:
            # Paths for marginal sampling contribution vectors
            root = osp.join(project_root, "models", dataset, METHOD, CLASSIFIER,
                           f"seed_{seed}", "interactions_marginal")
            contribution_root = osp.join(root, "contribution_vectors")
            compatibility_dir = osp.join(root, "compatibility")

            if not osp.exists(contribution_root):
                print(f"  [WARN] No contribution vectors for {dataset}/seed_{seed}")
                continue

            makedirs(compatibility_dir)

            # Load all contribution vectors
            contribution_vectors = {}
            for metric in METRICS:
                cv_path = osp.join(contribution_root, f"{metric}_interaction_contribution.npy")
                if osp.exists(cv_path):
                    contribution_vectors[metric] = np.load(cv_path)

            if not contribution_vectors:
                print(f"  [WARN] No contribution vectors loaded for {dataset}/seed_{seed}")
                continue

            # Calculate compatibility for each pair
            compatibility = {}
            for pair in metric_pairs:
                if pair[0] not in contribution_vectors or pair[1] not in contribution_vectors:
                    continue

                cv1 = contribution_vectors[pair[0]]
                cv2 = contribution_vectors[pair[1]]

                similarity, component_vector = cosine_similarity_components(cv1, cv2)

                pair_key = f"{pair[0]}-{pair[1]}"
                compatibility[pair_key] = similarity

                # Save component vector as _components.npy
                components_path = osp.join(compatibility_dir, f"{pair[0]}-{pair[1]}_components.npy")
                np.save(components_path, component_vector)
                total_files += 1

            # Save compatibility.json
            compat_path = osp.join(compatibility_dir, "compatibility.json")
            with open(compat_path, 'w') as f:
                json.dump(compatibility, f, indent=2)

            total_pairs += len(compatibility)
            print(f"  seed {seed}: {len(compatibility)} pairs, saved to {compatibility_dir}")

    print(f"\nTotal pairs processed: {total_pairs}")
    print(f"Total _components.npy files saved: {total_files}")
    print("\nDone! Now you can use plot4RQ2_order.py style logic for marginal sampling.")


if __name__ == "__main__":
    calculate_marginal_components()