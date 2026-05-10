"""
Analyze Baseline Robustness: Vector-level Consistency
Step 3: Compute cosine similarity between interaction vectors from different baselines.
Extended to 6 datasets: census, ufrgs, compas, diabetes, bank, heart
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import os.path as osp
import sys
import json
import numpy as np
from tqdm import tqdm

script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")

METRICS = ["accuracy", "recall", "FPR", "SPD", "EOD", "PED", "AOD", "CFVR", "GIFVR"]

BASELINES = {
    "mean": "interactions",
    "median": "interactions_median",
    "mode": "interactions_mode"
}

DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
METHOD = "vanilla"
CLASSIFIER = "lr"
SEEDS = list(range(10))


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1 = v1.flatten()
    v2 = v2.flatten()
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_contribution_vector(dataset, seed, baseline_suffix, metric):
    """Load a contribution vector for a given dataset, seed, baseline type, and metric."""
    path = osp.join(
        project_root, "models", dataset, METHOD, CLASSIFIER, 
        f"seed_{seed}", baseline_suffix, "contribution_vectors",
        f"{metric}_interaction_contribution.npy"
    )
    if not osp.exists(path):
        return None
    return np.load(path)


def compute_similarities_for_dataset(dataset):
    """Compute pairwise cosine similarities for all metrics across baselines for a single dataset."""
    results = {metric: {"mean_median": [], "mean_mode": [], "median_mode": []} for metric in METRICS}
    
    for seed in SEEDS:
        print(f"  Dataset {dataset}: Processing seed {seed}...")
        for metric in METRICS:
            vec_mean = load_contribution_vector(dataset, seed, BASELINES["mean"], metric)
            vec_median = load_contribution_vector(dataset, seed, BASELINES["median"], metric)
            vec_mode = load_contribution_vector(dataset, seed, BASELINES["mode"], metric)
            
            if vec_mean is None or vec_median is None or vec_mode is None:
                print(f"    Warning: Missing data for {metric} at seed {seed}")
                continue
            
            sim_mean_median = cosine_similarity(vec_mean, vec_median)
            sim_mean_mode = cosine_similarity(vec_mean, vec_mode)
            sim_median_mode = cosine_similarity(vec_median, vec_mode)
            
            results[metric]["mean_median"].append(sim_mean_median)
            results[metric]["mean_mode"].append(sim_mean_mode)
            results[metric]["median_mode"].append(sim_median_mode)
    
    return results


def compute_average_consistency(results):
    """Compute average consistency across all metric pairs."""
    avg_results = {}
    for metric in METRICS:
        mm = np.mean(results[metric]["mean_median"])
        mom = np.mean(results[metric]["mean_mode"])
        mdm = np.mean(results[metric]["median_mode"])
        avg = np.mean([mm, mom, mdm])
        avg_results[metric] = {
            "mean_median": mm,
            "mean_mode": mom,
            "median_mode": mdm,
            "avg": avg
        }
    return avg_results


def generate_markdown_table(all_dataset_results, output_path):
    """Generate markdown table with aggregated results across all datasets."""
    lines = []
    lines.append("# Baseline Robustness Analysis: Vector-level Consistency")
    lines.append("")
    lines.append("## Methodology")
    lines.append("- **Datasets**: Census Income, UFRGS, COMPAS, Diabetes, Bank Marketing, Heart Disease")
    lines.append("- **Method**: Vanilla + Logistic Regression")
    lines.append("- **Seeds**: 0-9 (10 random seeds per dataset, 60 total combinations)")
    lines.append("- **Baselines**: Mean (original), Median, Mode")
    lines.append("- **Metric**: Cosine Similarity of Interaction Vectors")
    lines.append("")
    lines.append("> **Interpretation**: Values > 0.9 (ideally > 0.95) indicate that the interaction vector")
    lines.append("> direction is robust to baseline selection, supporting the reliability of Harsanyi decomposition.")
    lines.append("")
    lines.append("## Results: Vector-level Consistency (Cosine Similarity)")
    lines.append("")
    lines.append("| Metric | Mean↔Median | Mean↔Mode | Median↔Mode | Avg. Consistency |")
    lines.append("| --- | --- | --- | --- | --- |")
    
    # Aggregate across all datasets
    agg_results = {metric: {"mean_median": [], "mean_mode": [], "median_mode": [], "avg": []} for metric in METRICS}
    
    for dataset, dataset_results in all_dataset_results.items():
        for metric in METRICS:
            agg_results[metric]["mean_median"].append(dataset_results[metric]["mean_median"])
            agg_results[metric]["mean_mode"].append(dataset_results[metric]["mean_mode"])
            agg_results[metric]["median_mode"].append(dataset_results[metric]["median_mode"])
            agg_results[metric]["avg"].append(dataset_results[metric]["avg"])
    
    # Compute final averages across all 60 combinations (6 datasets × 10 seeds)
    col_avgs = [0.0, 0.0, 0.0, 0.0]
    count = 0
    
    for metric in METRICS:
        mm = np.mean(agg_results[metric]["mean_median"])
        mom = np.mean(agg_results[metric]["mean_mode"])
        mdm = np.mean(agg_results[metric]["median_mode"])
        avg = np.mean([mm, mom, mdm])
        lines.append(f"| {metric} | {mm:.4f} | {mom:.4f} | {mdm:.4f} | {avg:.4f} |")
        col_avgs[0] += mm
        col_avgs[1] += mom
        col_avgs[2] += mdm
        col_avgs[3] += avg
        count += 1
    
    lines.append("| **Average** | **{:.4f}** | **{:.4f}** | **{:.4f}** | **{:.4f}** |".format(
        col_avgs[0]/count, col_avgs[1]/count, col_avgs[2]/count, col_avgs[3]/count))
    
    lines.append("")
    lines.append("---")
    lines.append("*Generated by baseline_robustness/analyze_baseline_robustness.py*")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("Baseline Robustness Analysis: Vector-level Consistency")
    print("=" * 60)
    print(f"Datasets: {DATASETS}")
    print(f"Method: {METHOD}, Classifier: {CLASSIFIER}")
    print(f"Seeds: {SEEDS} (10 per dataset)")
    print(f"Metrics: {METRICS}")
    print(f"Baselines: {list(BASELINES.keys())}")
    print("=" * 60)
    
    all_dataset_results = {}
    
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        results = compute_similarities_for_dataset(dataset)
        avg_results = compute_average_consistency(results)
        all_dataset_results[dataset] = avg_results
    
    # Print results
    print("\n" + "=" * 60)
    print("Vector-level Consistency (Cosine Similarity)")
    print("=" * 60)
    print(f"\n{'Metric':<12} {'Mean↔Median':>12} {'Mean↔Mode':>12} {'Median↔Mode':>12} {'Avg.':>12}")
    print("-" * 60)
    
    agg_results = {metric: {"mean_median": [], "mean_mode": [], "median_mode": [], "avg": []} for metric in METRICS}
    
    for dataset, dataset_results in all_dataset_results.items():
        for metric in METRICS:
            agg_results[metric]["mean_median"].append(dataset_results[metric]["mean_median"])
            agg_results[metric]["mean_mode"].append(dataset_results[metric]["mean_mode"])
            agg_results[metric]["median_mode"].append(dataset_results[metric]["median_mode"])
            agg_results[metric]["avg"].append(dataset_results[metric]["avg"])
    
    for metric in METRICS:
        mm = np.mean(agg_results[metric]["mean_median"])
        mom = np.mean(agg_results[metric]["mean_mode"])
        mdm = np.mean(agg_results[metric]["median_mode"])
        avg = np.mean(agg_results[metric]["avg"])
        print(f"{metric:<12} {mm:>12.4f} {mom:>12.4f} {mdm:>12.4f} {avg:>12.4f}")
    
    avg_mm = np.mean([np.mean(agg_results[m]["mean_median"]) for m in METRICS])
    avg_mom = np.mean([np.mean(agg_results[m]["mean_mode"]) for m in METRICS])
    avg_mdm = np.mean([np.mean(agg_results[m]["median_mode"]) for m in METRICS])
    avg_all = np.mean([np.mean(agg_results[m]["avg"]) for m in METRICS])
    print("-" * 60)
    print(f"{'Average':<12} {avg_mm:>12.4f} {avg_mom:>12.4f} {avg_mdm:>12.4f} {avg_all:>12.4f}")
    
    # Save detailed JSON results
    output_dir = osp.join(script_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = osp.join(output_dir, "vector_consistency_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    
    # Generate markdown table
    md_path = osp.join(output_dir, "vector_consistency_table.md")
    md_content = generate_markdown_table(all_dataset_results, md_path)
    print(f"Markdown table saved to: {md_path}")
    
    print("\n" + "=" * 60)
    print("Markdown Table Content:")
    print("=" * 60)
    print(md_content)


if __name__ == "__main__":
    main()
