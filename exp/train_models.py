import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import numpy as np
import joblib
import json
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback
import pandas as pd
import torch
torch.set_num_threads(4)

from tools.config import RANDOM_SEED_LIST, considered_sensitive_attributes, preprocessed_df_columns, privileged_groups, unprivileged_groups, MODELS, DATASETS
from tools.utils import set_seed, makedirs, NumpyEncoder
from tools.models import get_classifier, fit_classifier, save_classifier
from tools.evaluation import measure_final_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing


def train_single_vanilla_model(task_params):
    """
    Train a single model (worker function for parallel execution)
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        # Prepare data
        X_train_scaled = scaler.transform(data_train[:, :-1])
        X_test_scaled = scaler.transform(data_test[:, :-1])
        y_train = data_train[:, -1]
        y_test = data_test[:, -1]
        # sample_weights = compute_sample_weight('balanced', y_train)
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        model_save_root = osp.join(model_save_dir, classifier_name)
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] Training started...")
        
        # Get and train classifier
        clf = get_classifier(classifier_name, 
                           n_samples=data_train.shape[0], 
                           n_features=data_train.shape[1]-1, 
                           random_state=seed)
        
        if dataset_name == "census" and classifier_name == "tabnet":
            # clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train, 
            #                    sample_weight=sample_weights, 
            #                    save_checkpoints=True, 
            #                    checkpoint_path=model_save_root)
            clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train, 
                               save_checkpoints=True, 
                               checkpoint_path=model_save_root)
        else:
            # clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train, 
            #                    sample_weight=sample_weights)
            clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train)
        
        # Save model
        save_classifier(clf, classifier_name, model_save_root)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        def predict_func(X):
            X_scaled = scaler.transform(X)
            return clf.predict(X_scaled)
        
        eval_result = measure_final_score(data_test[:, :-1], y_test, y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'metrics': eval_result
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'error': error_msg
        }


def train_vanilla_models(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST, 
                       n_jobs=None, verbose=True):
    """
    Train vanilla models with parallel execution
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    """
    method_name = "vanilla"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_vanilla_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_vanilla_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        print(f"{'='*70}\n")
    
    return results


def train_single_reweighing_model(task_params):
    """
    Train a single model with Reweighing method (worker function for parallel execution)
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        sensitive_attribute_names = task_params['sensitive_attribute_names']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        # Prepare data
        X_train = data_train[:, :-1]
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(data_test[:, :-1])
        y_train = data_train[:, -1]
        y_test = data_test[:, -1]
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        model_save_root = osp.join(model_save_dir, classifier_name)
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Reweighing] Training started...")
        
        # Create AIF360 BinaryLabelDataset for training data
        # Combine features and labels
        train_data_with_label = np.column_stack([X_train, y_train])
        
        # Get feature names
        column_names = preprocessed_df_columns[dataset_name]
        
        # Create a dictionary for the dataset
        train_dict = {col: train_data_with_label[:, i] for i, col in enumerate(column_names)}
        
        # Create DataFrame (required by AIF360)
        train_df = pd.DataFrame(train_dict)
        
        # Create BinaryLabelDataset
        train_dataset = BinaryLabelDataset(
            df=train_df,
            label_names=['Probability'],
            protected_attribute_names=sensitive_attribute_names,
            favorable_label=1.0,
            unfavorable_label=0.0
        )
        
        # Apply Reweighing
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Reweighing] Applying Reweighing...")
        
        RW = Reweighing(unprivileged_groups=unprivileged_groups[dataset_name],
                       privileged_groups=privileged_groups[dataset_name])
        
        # Fit and transform the dataset
        train_dataset_transformed = RW.fit_transform(train_dataset)
        
        # Extract reweighed sample weights (only weights are changed, features and labels remain the same)
        reweighed_weights = train_dataset_transformed.instance_weights.ravel()
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Reweighing] "
              f"Weights range: [{reweighed_weights.min():.4f}, {reweighed_weights.max():.4f}], "
              f"Mean: {reweighed_weights.mean():.4f}")
        
        # Get and train classifier with reweighed weights
        clf = get_classifier(classifier_name, 
                           n_samples=data_train.shape[0], 
                           n_features=data_train.shape[1]-1, 
                           random_state=seed)
        
        clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train, 
                               sample_weight=reweighed_weights)
        
        # Save model
        save_classifier(clf, classifier_name, model_save_root)
        
        # Save reweighing weights for analysis
        np.save(osp.join(model_save_root + "_reweighing_weights.npy"), reweighed_weights)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        def predict_func(X):
            X_scaled = scaler.transform(X)
            return clf.predict(X_scaled)
        
        eval_result = measure_final_score(data_test[:, :-1], y_test, y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Reweighing] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'metrics': eval_result,
            'weight_stats': {
                'min': float(reweighed_weights.min()),
                'max': float(reweighed_weights.max()),
                'mean': float(reweighed_weights.mean()),
                'std': float(reweighed_weights.std())
            }
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Reweighing] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'error': error_msg
        }


def train_with_reweighing(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST,
                         n_jobs=None, verbose=True):
    """
    Train models with Reweighing method for fairness
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    """
    method_name = "reweighing"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        sensitive_attribute_names = list(considered_sensitive_attributes[dataset_name].keys())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices,
                    'sensitive_attribute_names': sensitive_attribute_names
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Reweighing Training Configuration:")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_reweighing_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_reweighing_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'method': method_name,
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Reweighing Training Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        # Print weight statistics for successful tasks
        if results['success'] and verbose:
            print(f"\nSample Weight Statistics (first 5 tasks):")
            for result in results['success'][:5]:
                if 'weight_stats' in result:
                    ws = result['weight_stats']
                    print(f"  {result['dataset']}|{result['classifier']}|seed={result['seed']}: "
                          f"min={ws['min']:.4f}, max={ws['max']:.4f}, mean={ws['mean']:.4f}, std={ws['std']:.4f}")
        
        print(f"{'='*70}\n")
    
    return results


def flip_sensitive_attributes(data_train, sensitive_indices, constraints):
    """
    Flip sensitive attributes in training data
    
    Parameters:
    -----------
    data_train : np.ndarray, shape (n_samples, n_features + 1)
        Training data with labels in last column
    sensitive_indices : list
        Indices of sensitive attributes
    constraints : np.ndarray, shape (n_features, 2)
        Min and max values for each feature
    
    Returns:
    --------
    data_flipped : np.ndarray
        Training data with flipped sensitive attributes
    """
    
    # Copy the data
    data_flipped = data_train.copy()
    
    # Flip each sensitive attribute
    for sens_idx in sensitive_indices:
        min_val = int(constraints[sens_idx, 0])
        max_val = int(constraints[sens_idx, 1])
        
        # Generate random values within the valid range
        n_samples = data_flipped.shape[0]
        flipped_values = np.random.randint(min_val, max_val + 1, size=n_samples)
        
        # Replace the sensitive attribute values
        data_flipped[:, sens_idx] = flipped_values
    
    return data_flipped


def train_single_flipping_model(task_params):
    """
    Train a single model with flipping-based retraining (worker function for parallel execution)
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] Flipping sensitive attributes...")
        
        # Flip sensitive attributes in training data
        data_train_flipped = flip_sensitive_attributes(data_train, sensitive_indices, 
                                                       constraints)
        
        # Combine original and flipped data
        data_train_combined = np.vstack([data_train, data_train_flipped])
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] "
              f"Training data: {data_train.shape[0]} → {data_train_combined.shape[0]} "
              f"(original + flipped)")
        
        # Prepare data
        X_train_combined = data_train_combined[:, :-1]
        y_train_combined = data_train_combined[:, -1]
        
        X_train_scaled = scaler.transform(X_train_combined)
        X_test_scaled = scaler.transform(data_test[:, :-1])
        y_test = data_test[:, -1]
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        model_save_root = osp.join(model_save_dir, classifier_name)
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] Training started...")
        
        # Get and train classifier
        clf = get_classifier(classifier_name, 
                           n_samples=data_train.shape[0], 
                           n_features=data_train.shape[1]-1, 
                           random_state=seed)
        
        clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train_combined)
        
        # Save model
        save_classifier(clf, classifier_name, model_save_root)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        def predict_func(X):
            X_scaled = scaler.transform(X)
            return clf.predict(X_scaled)
        
        eval_result = measure_final_score(data_test[:, :-1], y_test, y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'metrics': eval_result,
            'train_size_original': data_train.shape[0],
            'train_size_combined': data_train_combined.shape[0]
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'error': error_msg
        }


def train_flipping_models(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST, 
                         n_jobs=None, verbose=True):
    """
    Train models with flipping-based retraining using parallel execution
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    """
    method_name = "flipping"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Flipping-Based Retraining Configuration:")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"  Method: Flip sensitive attributes + append to original data")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_flipping_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_flipping_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Flipping-Based Retraining Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['success']:
            # Show some statistics about data augmentation
            sample_result = results['success'][0]
            if 'train_size_original' in sample_result:
                print(f"\nData Augmentation Info (sample):")
                print(f"  Original size: {sample_result['train_size_original']}")
                print(f"  Combined size: {sample_result['train_size_combined']}")
                print(f"  Augmentation ratio: 2x (original + flipped)")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        print(f"{'='*70}\n")
    
    return results


def train_single_blindness_model(task_params):
    """
    Train a single model with fairness through blindness (worker function for parallel execution)
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        # Create mask to exclude sensitive attributes
        n_features = data_train.shape[1] - 1  # Exclude label column
        mask = np.ones(n_features, dtype=bool)
        mask[sensitive_indices] = False
        
        # Apply mask to remove sensitive attributes
        X_train = data_train[:, :-1][:, mask]
        y_train = data_train[:, -1]
        
        # Scale data (fit scaler on non-sensitive features only)
        X_train_scaled = scaler.transform(data_train[:, :-1])[:, mask]
        X_test_scaled = scaler.transform(data_test[:, :-1])[:, mask]
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        model_save_root = osp.join(model_save_dir, classifier_name)
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Blindness] Training started...")
        print(f"  Original features: {n_features}, After removing sensitive: {X_train.shape[1]}")
        
        # Get and train classifier (with reduced feature dimension)
        clf = get_classifier(classifier_name, 
                           n_samples=X_train.shape[0], 
                           n_features=X_train.shape[1],  # Use reduced feature count
                           random_state=seed)
        
        clf = fit_classifier(clf, classifier_name, X_train_scaled, y_train)
        
        # Save model
        save_classifier(clf, classifier_name, model_save_root)
        
        # Save mask for inference
        mask_save_path = osp.join(model_save_dir, "sensitive_mask.npy")
        np.save(mask_save_path, mask)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Define predict function that applies mask
        def predict_func(X):
            for idx in sorted(sensitive_indices):
                X = np.insert(X, idx, 0, axis=1)
            X_scaled = scaler.transform(X)[:, mask]
            return clf.predict(X_scaled)
        
        # Evaluation uses original data (with sensitive attributes) for fairness metrics
        eval_result = measure_final_score(data_test[:, :-1], data_test[:, -1], y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, awareness="without_SA", seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Blindness] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'metrics': eval_result
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Blindness] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'error': error_msg
        }


def train_through_blindness(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST, 
                          n_jobs=None, verbose=True):
    """
    Train models with fairness through blindness (removing sensitive attributes)
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    """
    method_name = "blindness"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Fairness Through Blindness Training Configuration:")
        print(f"  Method: Remove sensitive attributes before training")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_blindness_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_blindness_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'method': method_name,
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Fairness Through Blindness Training Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        print(f"{'='*70}\n")
    
    return results


def generate_group_class_combinations(n_sensitive_attrs):
    """
    Generate all possible combinations of sensitive attributes and class labels
    
    Parameters:
    -----------
    n_sensitive_attrs : int
        Number of sensitive attributes
    
    Returns:
    --------
    combs : list of lists
        All binary combinations, e.g., [[0,0,0], [0,0,1], ..., [1,1,1]]
    """
    length = n_sensitive_attrs + 1  # +1 for class label
    combs = list(range(2 ** length))
    combs = [np.binary_repr(comb, width=length) for comb in combs]
    combs = [[int(item) for item in comb] for comb in combs]
    return combs


def get_nearest_neighbors(data_subset, knn, random_state=42):
    """
    Get a random sample and its nearest neighbors
    
    Parameters:
    -----------
    data_subset : np.ndarray
        Subset of data for a specific group
    knn : NearestNeighbors
        Fitted KNN model
    random_state : int
        Random seed
    
    Returns:
    --------
    parent, neighbor1, neighbor2 : np.ndarray
        Parent sample and two neighbors
    """
    np.random.seed(random_state)
    rand_idx = np.random.randint(0, data_subset.shape[0])
    parent = data_subset[rand_idx]
    
    # Get 3 nearest neighbors (including itself)
    ngbr_indices = knn.kneighbors(parent.reshape(1, -1), 3, return_distance=False)[0]
    
    neighbor1 = data_subset[ngbr_indices[1]]  # Skip index 0 (itself)
    neighbor2 = data_subset[ngbr_indices[2]]
    
    return parent, neighbor1, neighbor2


def generate_synthetic_samples(data_subset, n_samples, cr=0.8, f=0.8, random_state=42):
    """
    Generate synthetic samples using Fair-SMOTE strategy
    
    Parameters:
    -----------
    data_subset : np.ndarray, shape (n_subset, n_features)
        Data subset for oversampling
    n_samples : int
        Number of synthetic samples to generate
    cr : float
        Crossover rate (probability of mutation)
    f : float
        Mutation factor
    random_state : int
        Random seed
    
    Returns:
    --------
    synthetic_data : np.ndarray, shape (n_samples, n_features)
        Generated synthetic samples
    """
    if n_samples == 0:
        return np.array([]).reshape(0, data_subset.shape[1])
    
    if data_subset.shape[0] < 3:
        # If too few samples, just duplicate existing ones
        indices = np.random.choice(data_subset.shape[0], n_samples, replace=True)
        return data_subset[indices]
    
    # Fit KNN
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=min(3, data_subset.shape[0]), algorithm='auto')
    knn.fit(data_subset)
    
    synthetic_samples = []
    
    for i in range(n_samples):
        seed = random_state + i
        np.random.seed(seed)
        
        parent, child1, child2 = get_nearest_neighbors(data_subset, knn, seed)
        
        # Generate new sample using differential evolution
        new_sample = np.zeros_like(parent)
        
        for j in range(len(parent)):
            if np.random.random() < cr:
                # Apply mutation
                new_sample[j] = parent[j] + f * (child1[j] - child2[j])
            else:
                # Keep parent value
                new_sample[j] = parent[j]
        
        # Ensure non-negative values
        new_sample = np.abs(new_sample)
        
        synthetic_samples.append(new_sample)
    
    return np.array(synthetic_samples)


def apply_fairsmote_oversampling(data_train, sensitive_indices, random_state=42):
    """
    Apply Fair-SMOTE oversampling to balance all groups
    
    Parameters:
    -----------
    data_train : np.ndarray, shape (n_samples, n_features + 1)
        Training data with labels in last column
    sensitive_indices : list
        Indices of sensitive attributes
    random_state : int
        Random seed
    
    Returns:
    --------
    data_balanced : np.ndarray
        Balanced training data
    group_stats : dict
        Statistics about group sizes before and after balancing
    """
    n_sensitive = len(sensitive_indices)
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    
    # Generate all group-class combinations
    combs = generate_group_class_combinations(n_sensitive)
    
    # Split data into groups
    group_data = []
    group_sizes_before = []
    
    for comb in combs:
        # Create mask for this combination
        mask = (y_train == comb[-1])  # Class label
        for i, sens_idx in enumerate(sensitive_indices):
            mask = mask & (X_train[:, sens_idx] == comb[i])
        
        group_subset = data_train[mask]
        group_data.append(group_subset)
        group_sizes_before.append(len(group_subset))
    
    # Find maximum group size
    max_size = max(group_sizes_before)
    
    if max_size == 0:
        raise ValueError("All groups are empty!")
    
    # Oversample each group to max_size
    balanced_data = []
    group_sizes_after = []
    
    for i, (group_subset, original_size) in enumerate(zip(group_data, group_sizes_before)):
        if original_size == 0:
            # Skip empty groups
            continue
        
        n_to_generate = max_size - original_size
        
        if n_to_generate > 0:
            # Generate synthetic samples
            synthetic = generate_synthetic_samples(
                group_subset, 
                n_to_generate, 
                cr=0.8, 
                f=0.8, 
                random_state=random_state + i
            )
            group_balanced = np.vstack([group_subset, synthetic])
        else:
            group_balanced = group_subset
        
        balanced_data.append(group_balanced)
        group_sizes_after.append(len(group_balanced))
    
    # Combine all groups
    data_balanced = np.vstack(balanced_data)
    
    # Shuffle
    np.random.seed(random_state)
    shuffle_idx = np.random.permutation(len(data_balanced))
    data_balanced = data_balanced[shuffle_idx]
    
    # Statistics
    group_stats = {
        'n_groups': len(combs),
        'group_sizes_before': group_sizes_before,
        'group_sizes_after': group_sizes_after,
        'max_size': max_size,
        'total_before': sum(group_sizes_before),
        'total_after': len(data_balanced)
    }
    
    return data_balanced, group_stats


def situation_testing(clf, X_train, y_train, sensitive_indices, scaler, random_state=42):
    """
    Apply situation testing to remove discriminatory samples
    
    Parameters:
    -----------
    clf : classifier
        Trained classifier for situation testing
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sensitive_indices : list
        Indices of sensitive attributes
    scaler : StandardScaler
        Fitted scaler
    random_state : int
        Random seed
    
    Returns:
    --------
    X_fair : np.ndarray
        Features after removing discriminatory samples
    y_fair : np.ndarray
        Labels after removing discriminatory samples
    removal_stats : dict
        Statistics about removed samples
    """
    X_train_scaled = scaler.transform(X_train)
    
    # Get original predictions
    y_pred_original = clf.predict(X_train_scaled)
    
    # Create mask for samples that pass all tests
    pass_all_tests = np.ones(len(X_train), dtype=bool)
    
    flip_stats = {}
    
    for sens_idx in sensitive_indices:
        # Create flipped version
        X_flipped = X_train.copy()
        
        # Flip sensitive attribute (0 -> 1, 1 -> 0)
        X_flipped[:, sens_idx] = 1 - X_flipped[:, sens_idx]
        
        # Get predictions on flipped data
        X_flipped_scaled = scaler.transform(X_flipped)
        y_pred_flipped = clf.predict(X_flipped_scaled)
        
        # Find samples where prediction changed
        prediction_changed = (y_pred_original != y_pred_flipped)
        
        # Update mask (keep only samples that pass this test)
        pass_all_tests = pass_all_tests & (~prediction_changed)
        
        flip_stats[f'sens_attr_{sens_idx}'] = {
            'n_changed': int(prediction_changed.sum()),
            'pct_changed': float(prediction_changed.mean() * 100)
        }
    
    # Keep only samples that passed all tests
    X_fair = X_train[pass_all_tests]
    y_fair = y_train[pass_all_tests]
    
    removal_stats = {
        'n_original': len(X_train),
        'n_removed': int((~pass_all_tests).sum()),
        'n_remaining': len(X_fair),
        'pct_removed': float((~pass_all_tests).mean() * 100),
        'flip_stats': flip_stats
    }
    
    return X_fair, y_fair, removal_stats


def train_single_fairsmote_model(task_params):
    """
    Train a single model with Fair-SMOTE method (worker function for parallel execution)
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        model_save_root = osp.join(model_save_dir, classifier_name)
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] Training started...")
        
        # Step 1: Apply Fair-SMOTE oversampling
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] Applying oversampling...")
        data_oversampled, group_stats = apply_fairsmote_oversampling(
            data_train=data_train,
            sensitive_indices=sensitive_indices,
            random_state=seed
        )
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] "
              f"Oversampling: {group_stats['total_before']} → {group_stats['total_after']} samples")
        
        # Step 2: Train temporary classifier for situation testing
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] Training temporary RF for situation testing...")
        X_oversampled = data_oversampled[:, :-1]
        y_oversampled = data_oversampled[:, -1]
        
        X_oversampled_scaled = scaler.transform(X_oversampled)
        
        clf_temp = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=4)
        clf_temp.fit(X_oversampled_scaled, y_oversampled)
        
        # Step 3: Apply situation testing
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] Applying situation testing...")
        X_fair, y_fair, removal_stats = situation_testing(
            clf=clf_temp,
            X_train=X_oversampled,
            y_train=y_oversampled,
            sensitive_indices=sensitive_indices,
            scaler=scaler,
            random_state=seed
        )
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] "
              f"Situation testing: removed {removal_stats['n_removed']} ({removal_stats['pct_removed']:.2f}%) samples")
        
        # Step 4: Train final classifier on fair data
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] Training final classifier...")
        X_fair_scaled = scaler.transform(X_fair)
        X_test_scaled = scaler.transform(data_test[:, :-1])
        y_test = data_test[:, -1]
        
        clf = get_classifier(classifier_name, 
                           n_samples=len(X_fair), 
                           n_features=X_fair.shape[1], 
                           random_state=seed)
        
        clf = fit_classifier(clf, classifier_name, X_fair_scaled, y_fair)
        
        # Save model
        save_classifier(clf, classifier_name, model_save_root)
        
        # Save Fair-SMOTE statistics
        fairsmote_stats = {
            'group_stats': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in group_stats.items()},
            'removal_stats': removal_stats
        }
        with open(osp.join(model_save_root + "_fairsmote_stats.json"), "w") as f:
            json.dump(fairsmote_stats, f, indent=1, cls=NumpyEncoder)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        def predict_func(X):
            X_scaled = scaler.transform(X)
            return clf.predict(X_scaled)
        
        eval_result = measure_final_score(data_test[:, :-1], y_test, y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'metrics': eval_result,
            'fairsmote_stats': fairsmote_stats
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|Fair-SMOTE] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'error': error_msg
        }


def train_with_fairsmote(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST,
                         n_jobs=None, verbose=True):
    """
    Train models with Fair-SMOTE method for fairness
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    """
    method_name = "fairsmote"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Fair-SMOTE Training Configuration:")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_fairsmote_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_fairsmote_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'method': method_name,
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Fair-SMOTE Training Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        # Print Fair-SMOTE statistics for successful tasks
        if results['success'] and verbose:
            print(f"\nFair-SMOTE Statistics (first 5 tasks):")
            for result in results['success'][:5]:
                if 'fairsmote_stats' in result:
                    gs = result['fairsmote_stats']['group_stats']
                    rs = result['fairsmote_stats']['removal_stats']
                    print(f"  {result['dataset']}|{result['classifier']}|seed={result['seed']}:")
                    print(f"    Oversampling: {gs['total_before']} → {gs['total_after']}")
                    print(f"    Removed: {rs['n_removed']} ({rs['pct_removed']:.2f}%)")
        
        print(f"{'='*70}\n")
    
    return results


def training_data_debugging(data_train, sensitive_idx, constraints):
    """
    Perform training data debugging for fairness model
    
    Implements undersampling by reducing samples in the Privileged & Favorable 
    and Unprivileged & Unfavorable subgroups to alleviate label bias and selection bias.
    
    Parameters:
    -----------
    data_train : np.ndarray, shape (n_samples, n_features + 1)
        Training data with labels in last column
    sensitive_idx : int
        Index of the sensitive attribute
    constraints : np.ndarray, shape (n_features, 2)
        Min and max values for each feature
    
    Returns:
    --------
    data_debugged : np.ndarray
        Debugged training data after undersampling
    """
    
    # Extract sensitive attribute and labels
    sensitive_attr = data_train[:, sensitive_idx]
    labels = data_train[:, -1]
    
    # Count samples in four subgroups
    # zero_zero: Unprivileged (0) & Unfavorable (0)
    # zero_one: Privileged (1) & Unfavorable (0)
    # one_zero: Unprivileged (0) & Favorable (1)
    # one_one: Privileged (1) & Favorable (1)
    zero_zero = np.sum((sensitive_attr == 0) & (labels == 0))
    zero_one = np.sum((sensitive_attr == 1) & (labels == 0))
    one_zero = np.sum((sensitive_attr == 0) & (labels == 1))
    one_one = np.sum((sensitive_attr == 1) & (labels == 1))
    
    # Calculate undersampling amounts using quadratic formula
    # Solving for x and y to satisfy WAE (We're All Equal) criteria
    import math
    
    a = zero_one + one_one
    b = -1 * (zero_zero * zero_one + 2 * zero_zero * one_one + one_zero * one_one)
    c = (zero_zero + one_zero) * (zero_zero * one_one - zero_one * one_zero)
    
    # Handle edge cases
    discriminant = b * b - 4 * a * c
    if discriminant < 0 or a == 0:
        # If no valid solution, return original data
        return data_train
    
    x = (-b - math.sqrt(discriminant)) / (2 * a)
    y = (zero_one + one_one) / (zero_zero + one_zero) * x if (zero_zero + one_zero) > 0 else 0
    
    # Calculate new sample sizes
    zero_zero_new = int(max(0, zero_zero - x))
    one_one_new = int(max(0, one_one - y))
    
    # Get indices for each subgroup
    idx_zero_zero = np.where((sensitive_attr == 0) & (labels == 0))[0]
    idx_zero_one = np.where((sensitive_attr == 1) & (labels == 0))[0]
    idx_one_zero = np.where((sensitive_attr == 0) & (labels == 1))[0]
    idx_one_one = np.where((sensitive_attr == 1) & (labels == 1))[0]
    
    # Perform undersampling
    if zero_zero_new < zero_zero and zero_zero_new > 0:
        idx_zero_zero = np.random.choice(idx_zero_zero, size=zero_zero_new, replace=False)
    
    if one_one_new < one_one and one_one_new > 0:
        idx_one_one = np.random.choice(idx_one_one, size=one_one_new, replace=False)
    
    # Combine all indices
    selected_indices = np.concatenate([idx_zero_zero, idx_zero_one, idx_one_zero, idx_one_one])
    
    # Shuffle and return
    np.random.shuffle(selected_indices)
    data_debugged = data_train[selected_indices]
    
    return data_debugged


def train_single_maat_model(task_params):
    """
    Train a single MAAT model (worker function for parallel execution)
    
    MAAT combines fairness models (one per sensitive attribute) and a performance model
    using ensemble learning to improve fairness-performance trade-off.
    
    Parameters:
    -----------
    task_params : dict
        Dictionary containing all parameters needed for training
    
    Returns:
    --------
    dict : Training result with status and metrics
    """
    try:
        # Unpack parameters
        dataset_name = task_params['dataset_name']
        classifier_name = task_params['classifier_name']
        seed = task_params['seed']
        method_name = task_params['method_name']
        sensitive_indices = task_params['sensitive_indices']
        
        # Set seed for this process
        set_seed(seed)
        
        # Load data
        data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
        data_train = np.load(osp.join(data_save_root, "data_train.npy"))
        data_test = np.load(osp.join(data_save_root, "data_test.npy"))
        constraints = np.load(osp.join(data_save_root, "constraints.npy"))
        scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
        
        # Setup model save path
        model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
        makedirs(model_save_dir)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] Training started...")
        
        # ==================== Step 1: Train Fairness Models ====================
        fairness_models = []
        fairness_model_paths = []
        
        for i, sens_idx in enumerate(sensitive_indices):
            print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
                  f"Training fairness model {i+1}/{len(sensitive_indices)} for sensitive attribute {sens_idx}...")
            
            # Debug training data for this sensitive attribute
            data_train_debugged = training_data_debugging(data_train, sens_idx, constraints)
            
            print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
                  f"Data debugging: {data_train.shape[0]} → {data_train_debugged.shape[0]} samples")
            
            # Prepare data
            X_train_debugged = data_train_debugged[:, :-1]
            y_train_debugged = data_train_debugged[:, -1]
            X_train_scaled = scaler.transform(X_train_debugged)
            
            # Train fairness model
            fairness_clf = get_classifier(classifier_name, 
                                        n_samples=data_train_debugged.shape[0], 
                                        n_features=data_train_debugged.shape[1]-1, 
                                        random_state=seed)
            
            fairness_clf = fit_classifier(fairness_clf, classifier_name, 
                                        X_train_scaled, y_train_debugged)
            
            # Save fairness model
            fairness_model_path = osp.join(model_save_dir, f"{classifier_name}_fairness_{i}")
            save_classifier(fairness_clf, classifier_name, fairness_model_path)
            
            fairness_models.append(fairness_clf)
            fairness_model_paths.append(fairness_model_path)
            
            print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
                  f"Fairness model {i+1} saved to {fairness_model_path}")
        
        # ==================== Step 2: Train Performance Model ====================
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] Training performance model...")
        
        X_train = data_train[:, :-1]
        y_train = data_train[:, -1]
        X_train_scaled = scaler.transform(X_train)
        
        performance_clf = get_classifier(classifier_name, 
                                       n_samples=data_train.shape[0], 
                                       n_features=data_train.shape[1]-1, 
                                       random_state=seed)
        
        performance_clf = fit_classifier(performance_clf, classifier_name, 
                                       X_train_scaled, y_train)
        
        # Save performance model
        performance_model_path = osp.join(model_save_dir, f"{classifier_name}_performance")
        save_classifier(performance_clf, classifier_name, performance_model_path)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
              f"Performance model saved to {performance_model_path}")
        
        # ==================== Step 3: Ensemble Prediction ====================
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] Making ensemble predictions...")
        
        X_test = data_test[:, :-1]
        y_test = data_test[:, -1]
        X_test_scaled = scaler.transform(X_test)
        
        # Collect predictions from all models
        all_pred_probas = []
        
        # Fairness models predictions
        for i, fairness_clf in enumerate(fairness_models):
            pred_proba = fairness_clf.predict_proba(X_test_scaled)[:, 1]
            all_pred_probas.append(pred_proba)
            print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
                  f"Fairness model {i+1} prediction range: [{pred_proba.min():.4f}, {pred_proba.max():.4f}]")
        
        # Performance model prediction
        perf_pred_proba = performance_clf.predict_proba(X_test_scaled)[:, 1]
        all_pred_probas.append(perf_pred_proba)
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
              f"Performance model prediction range: [{perf_pred_proba.min():.4f}, {perf_pred_proba.max():.4f}]")
        
        # Stack predictions
        all_pred_probas = np.column_stack(all_pred_probas)
        
        # Average ensemble strategy (default 0.5-0.5 combination)
        ensemble_pred_proba = np.mean(all_pred_probas, axis=1)
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] "
              f"Ensemble prediction range: [{ensemble_pred_proba.min():.4f}, {ensemble_pred_proba.max():.4f}]")
        
        # ==================== Step 4: Evaluation ====================
        def predict_func(X):
            """Prediction function for counterfactual fairness evaluation"""
            X_scaled = scaler.transform(X)
            
            # Collect predictions from all models
            pred_probas = []
            
            # Fairness models
            for fairness_clf in fairness_models:
                pred_probas.append(fairness_clf.predict_proba(X_scaled)[:, 1])
            
            # Performance model
            pred_probas.append(performance_clf.predict_proba(X_scaled)[:, 1])
            
            # Ensemble
            pred_probas = np.column_stack(pred_probas)
            ensemble_proba = np.mean(pred_probas, axis=1)
            
            return (ensemble_proba >= 0.5).astype(int)
        
        eval_result = measure_final_score(X_test, y_test, ensemble_pred, 
                                         sensitive_indices, constraints, 
                                         ensemble_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        metrics_path = osp.join(model_save_dir, f"{classifier_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
        
        # Save ensemble information
        ensemble_info = {
            'n_fairness_models': len(fairness_models),
            'fairness_model_paths': fairness_model_paths,
            'performance_model_path': performance_model_path,
            'sensitive_indices': sensitive_indices,
            'ensemble_strategy': 'average',
            'train_size_original': data_train.shape[0],
            'train_sizes_debugged': [training_data_debugging(data_train, idx, constraints).shape[0] 
                                    for idx in sensitive_indices]
        }
        
        ensemble_info_path = osp.join(model_save_dir, f"{classifier_name}_ensemble_info.json")
        with open(ensemble_info_path, "w") as f:
            json.dump(ensemble_info, f, indent=1, cls=NumpyEncoder)
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] ✓ Completed")
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'metrics': eval_result,
            'ensemble_info': ensemble_info
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|MAAT] ✗ Failed: {str(e)}")
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'classifier': classifier_name,
            'seed': seed,
            'method': method_name,
            'error': error_msg
        }


def train_with_maat(dataset_name_list, classifier_name_list, seed_list=RANDOM_SEED_LIST,
                   n_jobs=None, verbose=True):
    """
    Train models with MAAT (fairness-performance ensemble) method
    
    MAAT combines multiple fairness models (one per sensitive attribute) with a 
    performance model to achieve better fairness-performance trade-off.
    
    Parameters:
    -----------
    dataset_name_list : list
        List of dataset names
    classifier_name_list : list
        List of classifier names
    seed_list : list
        List of random seeds
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all CPU cores.
        Set to 1 for sequential execution.
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict : Summary of training results
    
    References:
    -----------
    Chen et al. "MAAT: A Novel Ensemble Approach to Addressing Fairness and 
    Performance Bugs for Machine Learning Software." ESEC/FSE 2022.
    """
    method_name = "maat"
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = cpu_count() // 8
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Prepare all tasks
    tasks = []
    for dataset_name in dataset_name_list:
        sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
        
        for seed in seed_list:
            for classifier_name in classifier_name_list:
                task_params = {
                    'dataset_name': dataset_name,
                    'classifier_name': classifier_name,
                    'seed': seed,
                    'method_name': method_name,
                    'sensitive_indices': sensitive_indices
                }
                tasks.append(task_params)
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MAAT Training Configuration:")
        print(f"  Datasets: {dataset_name_list}")
        print(f"  Classifiers: {classifier_name_list}")
        print(f"  Seeds: {seed_list}")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Parallel workers: {n_jobs}")
        print(f"  Method: Fairness-Performance Ensemble (MAAT)")
        print(f"{'='*70}\n")
    
    # Execute tasks in parallel
    results = {
        'success': [],
        'failed': []
    }
    
    if n_jobs == 1:
        # Sequential execution
        if verbose:
            print("Running in sequential mode...")
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\nTask {i}/{total_tasks}")
            result = train_single_maat_model(task)
            if result['status'] == 'success':
                results['success'].append(result)
            else:
                results['failed'].append(result)
    else:
        # Parallel execution
        if verbose:
            print(f"Running in parallel mode with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_single_maat_model, task): task 
                            for task in tasks}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results['success'].append(result)
                    else:
                        results['failed'].append(result)
                except Exception as e:
                    error_result = {
                        'status': 'failed',
                        'dataset': task['dataset_name'],
                        'classifier': task['classifier_name'],
                        'seed': task['seed'],
                        'method': method_name,
                        'error': str(e)
                    }
                    results['failed'].append(error_result)
                    print(f"Task failed with exception: {e}")
                
                if verbose:
                    print(f"Progress: {completed}/{total_tasks} tasks completed")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"MAAT Training Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for fail in results['failed']:
                print(f"  - {fail['dataset']} | {fail['classifier']} | seed={fail['seed']}")
                print(f"    Error: {fail['error'][:100]}...")
        
        # Print ensemble statistics for successful tasks
        if results['success'] and verbose:
            print(f"\nEnsemble Statistics (first 5 tasks):")
            for result in results['success'][:5]:
                if 'ensemble_info' in result:
                    ei = result['ensemble_info']
                    print(f"  {result['dataset']}|{result['classifier']}|seed={result['seed']}:")
                    print(f"    - Fairness models: {ei['n_fairness_models']}")
                    print(f"    - Original train size: {ei['train_size_original']}")
                    print(f"    - Debugged train sizes: {ei['train_sizes_debugged']}")
        
        print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    train_vanilla_models(DATASETS, MODELS)
    train_with_reweighing(DATASETS, MODELS)
    train_flipping_models(DATASETS, MODELS)
    train_through_blindness(DATASETS, MODELS)
    train_with_fairsmote(DATASETS, MODELS)
    train_with_maat(DATASETS, MODELS)