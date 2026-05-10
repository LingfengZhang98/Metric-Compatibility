import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
import os.path as osp

from scipy import stats
import random
import json
import numpy as np
import torch
import itertools
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def set_seed(seed=0):
    print(f"Set SEED: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.random.set_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def max_output_diff_pair(similar_x, model):
    """
    Select a couple of instances such that the DNN outputs on them are maximally different.
    """
    outputs = model(similar_x)
    min_index = np.argmin(outputs)
    max_index = np.argmax(outputs)
    return similar_x[min_index], similar_x[max_index]


def enumerate_subgroups(sensitive_indices, value_ranges):
    # Get all possible combinations of sensitive attribute values
    sensitive_value_combinations = []
    for idx in sensitive_indices:
        min_val, max_val = value_ranges[idx]
        sensitive_value_combinations.append(list(range(min_val, max_val + 1)))

    # Generate all combinations
    return list(itertools.product(*sensitive_value_combinations))


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two 1D numpy arrays
    
    Parameters:
        a: 1D numpy array
        b: 1D numpy array with the same length as a
    
    Returns:
        similarity: cosine similarity scalar
        component_vector: component vector of cosine similarity
    """
    # Check if either input is a zero vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        # If either vector is zero, return 0 and zero vector
        return 0.0, np.zeros_like(a)
    
    # Calculate component vector: (a * b) / (||a|| * ||b||)
    component_vector = (a * b) / (norm_a * norm_b)
    
    # Cosine similarity scalar is the sum of component vector
    similarity = np.sum(component_vector)
    
    return similarity, component_vector


def calculate_mean_and_ci(data, confidence=0.95):
    """
    Calculate mean and confidence interval (t-distribution) for a list of data
    
    Parameters:
        data: list or array-like, input data
        confidence: float, confidence level, default 0.95 (95% confidence interval)
    
    Returns:
        dict: dictionary containing mean, lower bound, and upper bound of CI
    """
    # Convert to numpy array
    data = np.array(data)
    
    # Sample size
    n = len(data)
    
    # Calculate mean
    mean = np.mean(data)
    
    # Calculate standard error
    std_err = stats.sem(data)  # standard error = std / sqrt(n)
    
    # Calculate t-value (degrees of freedom = n-1)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    # Calculate confidence interval
    margin_of_error = t_value * std_err
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return {
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin_of_error': margin_of_error
    }


def calculate_mean_ci_for_correlations(correlations, confidence=0.95):
    """
    Calculate mean and confidence interval for correlation coefficients across datasets
    using Fisher z-transformation
    
    Parameters:
        correlations: list or array-like, correlation coefficients from different datasets
        confidence: float, confidence level, default 0.95 (95% CI)
    
    Returns:
        dict: dictionary containing mean correlation and CI bounds
    """
    # Convert to numpy array
    correlations = np.array(correlations)
    k = len(correlations)  # number of datasets
    
    # Step 1: Fisher z-transformation (arctanh)
    correlations_clipped = np.clip(correlations, -0.9999, 0.9999)   # Avoid arctanh(±1) = ±∞
    z_values = np.arctanh(correlations_clipped)
    
    # Step 2: Calculate mean and standard error in z-space
    z_mean = np.mean(z_values)
    z_std = np.std(z_values, ddof=1)  # sample standard deviation
    z_se = z_std / np.sqrt(k)  # standard error across datasets
    
    # Step 3: Calculate confidence interval in z-space using t-distribution
    df = k - 1  # degrees of freedom
    t_value = stats.t.ppf((1 + confidence) / 2, df)
    
    z_ci_lower = z_mean - t_value * z_se
    z_ci_upper = z_mean + t_value * z_se
    
    # Step 4: Transform back to correlation space using tanh
    mean_correlation = np.tanh(z_mean)
    ci_lower = np.tanh(z_ci_lower)
    ci_upper = np.tanh(z_ci_upper)
    
    return {
        'mean_correlation': mean_correlation,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_datasets': k
    }