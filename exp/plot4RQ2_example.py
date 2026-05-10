import os
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from scipy import stats

from tools.utils import makedirs
from tools.config import (DATASETS, MODELS, RANDOM_SEED_LIST, 
                          preprocessed_df_columns, pairs_to_analyze)


def try_read_with_order_invariance(base_path, metric1, metric2):
    """Try to read file with order invariance"""
    path1 = osp.join(base_path, f"{metric1}-{metric2}_components.npy")
    if osp.exists(path1):
        return np.load(path1), f"{metric1}-{metric2}"
    
    path2 = osp.join(base_path, f"{metric2}-{metric1}_components.npy")
    if osp.exists(path2):
        return np.load(path2), f"{metric2}-{metric1}"
    
    return None, None


def read_compatibility_with_order_invariance(compatibility_dict, metric1, metric2):
    """Read compatibility value with order invariance from loaded json dict"""
    key1 = f"{metric1}-{metric2}"
    if key1 in compatibility_dict:
        return compatibility_dict[key1]
    
    key2 = f"{metric2}-{metric1}"
    if key2 in compatibility_dict:
        return compatibility_dict[key2]
    
    return None


def mask_to_attribute_names(mask, attribute_names):
    """Convert a boolean mask to attribute combination names"""
    selected_attrs = [attr for attr, selected in zip(attribute_names, mask) if selected]
    return tuple(selected_attrs)


def read_data_for_pairs(pairs_to_analyze):
    """Read components and compatibility for specified pairs"""
    data_by_dataset = {}
    total_combinations = 0
    successful_reads = 0
    
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        
        data_by_dataset[dataset] = {'masks': None}
        
        for pair in pairs_to_analyze:
            pair_key = f"{pair[0]}-{pair[1]}"
            data_by_dataset[dataset][pair_key] = {
                'components': [],
                'compatibility': []
            }
        
        for classifier in MODELS:
            for seed in RANDOM_SEED_LIST:
                total_combinations += 1
                
                masks_path = osp.join(
                    script_dir, "../models", dataset, "vanilla", 
                    classifier, f"seed_{seed}", "interactions", "masks.npy"
                )
                
                if not osp.exists(masks_path):
                    continue
                
                masks = np.load(masks_path)
                
                if data_by_dataset[dataset]['masks'] is None:
                    data_by_dataset[dataset]['masks'] = masks
                
                compatibility_path = osp.join(
                    script_dir, "../models", dataset, "vanilla", 
                    classifier, f"seed_{seed}", "interactions",
                    "compatibility", "compatibility.json"
                )
                
                if not osp.exists(compatibility_path):
                    continue
                
                with open(compatibility_path, 'r') as f:
                    compatibility_dict = json.load(f)
                
                components_base_path = osp.join(
                    script_dir, "../models", dataset, "vanilla", 
                    classifier, f"seed_{seed}", "interactions", "compatibility"
                )
                
                pair_success = True
                for pair in pairs_to_analyze:
                    metric1, metric2 = pair
                    
                    components, _ = try_read_with_order_invariance(
                        components_base_path, metric1, metric2
                    )
                    
                    if components is None:
                        pair_success = False
                        break
                    
                    compatibility = read_compatibility_with_order_invariance(
                        compatibility_dict, metric1, metric2
                    )
                    
                    if compatibility is None:
                        pair_success = False
                        break
                    
                    pair_key = f"{pair[0]}-{pair[1]}"
                    data_by_dataset[dataset][pair_key]['components'].append(components)
                    data_by_dataset[dataset][pair_key]['compatibility'].append(compatibility)
                
                if pair_success:
                    successful_reads += 1
    
    print(f"\n{'='*60}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful reads: {successful_reads}")
    print(f"{'='*60}")
    
    return data_by_dataset


def compute_statistics(data_by_dataset):
    """Compute mean compatibility, mean contribution ratio, and top/bottom attribute combinations"""
    results = defaultdict(dict)
    
    for dataset in DATASETS:
        print(f"\nComputing statistics for dataset: {dataset}")
        
        attribute_names = preprocessed_df_columns[dataset][:-1]
        masks = data_by_dataset[dataset].get('masks')
        
        if masks is None:
            print(f"  Warning: No masks found for {dataset}")
            continue
        
        print(f"  Masks type: {type(masks)}, shape: {masks.shape}")
        
        for pair in pairs_to_analyze:
            pair_key = f"{pair[0]}-{pair[1]}"
            
            if pair_key not in data_by_dataset[dataset]:
                continue
            
            components_list = data_by_dataset[dataset][pair_key]['components']
            compatibility_list = data_by_dataset[dataset][pair_key]['compatibility']
            
            if len(components_list) == 0:
                continue
            
            components_array = np.array(components_list)
            compatibility_array = np.array(compatibility_list)
            
            mean_compat = np.mean(compatibility_array)
            sem_compat = stats.sem(compatibility_array)
            ci_compat = stats.t.interval(0.95, len(compatibility_array)-1, 
                                         loc=mean_compat, scale=sem_compat)
            
            mean_components = np.mean(components_array, axis=0)
            mean_contribution_ratio = mean_components / mean_compat if mean_compat != 0 else mean_components
            mean_contribution_percentage = mean_contribution_ratio * 100
            
            sorted_indices = np.argsort(mean_contribution_percentage)
            bottom_3_indices = sorted_indices[:3]
            top_3_indices = sorted_indices[-3:][::-1]
            
            top_attrs = []
            for idx in top_3_indices:
                mask = masks[idx]
                attrs = mask_to_attribute_names(mask, attribute_names)
                percentage = mean_contribution_percentage[idx]
                top_attrs.append((attrs, percentage))
            
            bottom_attrs = []
            for idx in bottom_3_indices:
                mask = masks[idx]
                attrs = mask_to_attribute_names(mask, attribute_names)
                percentage = mean_contribution_percentage[idx]
                bottom_attrs.append((attrs, percentage))
            
            results[dataset][pair_key] = {
                'mean_compat': mean_compat,
                'ci_compat': ci_compat,
                'top_attrs': top_attrs,
                'bottom_attrs': bottom_attrs,
                'n_samples': len(compatibility_list)
            }
            
            print(f"  {pair_key}: mean_compat={mean_compat:.4f}, n={len(compatibility_list)}")
    
    return results


def format_attribute_combination(attrs, max_length=None):
    """Format attribute combination for display"""
    if len(attrs) == 0:
        return "∅"
    
    attrs_str = ", ".join(attrs)
    
    if max_length is not None and len(attrs_str) > max_length:
        return attrs_str[:max_length-3] + "..."
    
    return attrs_str


def create_results_table(results):
    """Create a formatted CSV table with results"""
    rows = []
    
    for dataset in DATASETS:
        row_data = {'Dataset': dataset}
        
        for pair in pairs_to_analyze:
            pair_key = f"{pair[0]}-{pair[1]}"
            
            if dataset not in results or pair_key not in results[dataset]:
                row_data[f'{pair_key}_Mean Compatibility'] = 'N/A'
                row_data[f'{pair_key}_Top 3 Combinations'] = 'N/A'
                row_data[f'{pair_key}_Bottom 3 Combinations'] = 'N/A'
                continue
            
            data = results[dataset][pair_key]
            
            mean_compat = data['mean_compat']
            ci_low, ci_high = data['ci_compat']
            ci_margin = (ci_high - ci_low) / 2
            compat_str = f"{mean_compat:.4f} ± {ci_margin:.4f}"
            row_data[f'{pair_key}_Mean Compatibility'] = compat_str
            
            top_str_list = []
            for attrs, percentage in data['top_attrs']:
                attrs_str = format_attribute_combination(attrs)
                top_str_list.append(f"{attrs_str} ({percentage:.2f}%)")
            row_data[f'{pair_key}_Top 3 Combinations'] = " | ".join(top_str_list)
            
            bottom_str_list = []
            for attrs, percentage in data['bottom_attrs']:
                attrs_str = format_attribute_combination(attrs)
                bottom_str_list.append(f"{attrs_str} ({percentage:.2f}%)")
            row_data[f'{pair_key}_Bottom 3 Combinations'] = " | ".join(bottom_str_list)
        
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    
    ordered_columns = ['Dataset']
    for pair in pairs_to_analyze:
        pair_key = f"{pair[0]}-{pair[1]}"
        ordered_columns.extend([
            f'{pair_key}_Mean Compatibility',
            f'{pair_key}_Top 3 Combinations',
            f'{pair_key}_Bottom 3 Combinations'
        ])
    
    ordered_columns = [col for col in ordered_columns if col in df.columns]
    df = df[ordered_columns]
    
    return df


def save_results_to_csv(df, output_path):
    """Save results DataFrame to CSV with proper formatting"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    print("="*60)
    print("Analyzing Metric Pair Compatibility and Contributions")
    print("="*60)
    
    print("\n[Step 1] Reading data...")
    data_by_dataset = read_data_for_pairs(pairs_to_analyze)
    
    print("\n[Step 2] Computing statistics...")
    results = compute_statistics(data_by_dataset)
    
    print("\n[Step 3] Creating results table...")
    df = create_results_table(results)
    
    results_dir = osp.join(script_dir, "results", "RQ2")
    makedirs(results_dir)
    output_path = osp.join(results_dir, 'example_analysis.csv')
    save_results_to_csv(df, output_path)
    
    print("\n" + "="*60)
    print("Results Preview:")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=False))
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)