"""
Calculate Median and Mode baselines for ALL datasets.
Baseline Robustness Validation for Rebuttal.
Extended to 6 datasets: census, ufrgs, compas, diabetes, bank, heart
"""

import os
import os.path as osp
import sys
import numpy as np
from scipy import stats

script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")

SEED_LIST = list(range(10))
DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]


def compute_mode(arr):
    """Compute mode for each column. If multiple modes, return the first one."""
    mode_vals = []
    for col in arr.T:
        unique_vals, counts = np.unique(col, return_counts=True)
        max_count = counts.max()
        modes = unique_vals[counts == max_count]
        mode_vals.append(modes[0])
    return np.array(mode_vals, dtype=arr.dtype)


def compute_median(arr):
    """Compute median for each column. Use lower median (n//2) to keep integer."""
    median_vals = []
    n = arr.shape[0]
    mid_idx = n // 2
    for col in arr.T:
        sorted_col = np.sort(col)
        median_vals.append(sorted_col[mid_idx])
    return np.array(median_vals, dtype=arr.dtype)


def main():
    for DATASET in DATASETS:
        print("=" * 60)
        print(f"Processing dataset: {DATASET}")
        print("=" * 60)
        
        for seed in SEED_LIST:
            data_dir = osp.join(project_root, "data", "tabular", DATASET, "prepared_data", f"seed_{seed}")
            output_dir = osp.join(script_dir, DATASET, f"seed_{seed}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            data_train = np.load(osp.join(data_dir, "data_train.npy"))
            X_train = data_train[:, :-1]
            
            print(f"Seed {seed}: X_train shape = {X_train.shape}, dtype = {X_train.dtype}")
            print(f"  Sample values (first row): {X_train[0]}")
            
            mode_baseline = compute_mode(X_train)
            median_baseline = compute_median(X_train)
            
            np.save(osp.join(output_dir, "mode_baseline.npy"), mode_baseline)
            np.save(osp.join(output_dir, "median_baseline.npy"), median_baseline)
            
            print(f"  Mode baseline: {mode_baseline}")
            print(f"  Median baseline: {median_baseline}")
            
            def check_integer(arr, name):
                is_int = np.all(arr == arr.astype(int))
                print(f"  {name} all integers: {is_int}")
                if not is_int:
                    non_int_mask = arr != arr.astype(int)
                    print(f"    Non-integer values: {arr[non_int_mask]}")
                return is_int
            
            check_integer(mode_baseline, "Mode")
            check_integer(median_baseline, "Median")
            print()
        print()


if __name__ == "__main__":
    main()
