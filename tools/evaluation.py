"""
This comprehensive toolkit evaluates both model performance and fairness for binary classification tasks.
It implements 6 performance metrics (Accuracy, Recall, Precision, F1-Score, MCC, ROC-AUC)
and 6 fairness indicators including 4 group fairness metrics (SPD, EOD, PED, AOD)
and 2 individual fairness metrics (Causal Fairness and Global Individual Fairness).
The toolkit supports multiple sensitive attribute scenarios.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
import itertools

from .utils import enumerate_subgroups


def calculate_fpr(y_true, y_pred):
    """Calculate False Positive Rate"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def calculate_spd(y_pred, sensitive_data, sensitive_indices, value_ranges):
    """
    Calculate Statistical Parity Difference (SPD)
    SPD = max_s P[Y_hat=1|A=s] - min_s P[Y_hat=1|A=s]
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    for combination in all_combinations:
        # Create mask for current subgroup
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])

        # Calculate P[Y_hat=1|A=s]
        if np.sum(mask) > 0:
            prob = np.sum(y_pred[mask] == 1) / np.sum(mask)
            probabilities.append(prob)

    if len(probabilities) == 0:
        return 0.0

    return max(probabilities) - min(probabilities)


def calculate_eod(y_true, y_pred, sensitive_data, sensitive_indices, value_ranges):
    """
    Calculate Equalized Odds Difference (EOD)
    EOD = max_s P[Y_hat=1|A=s,Y=1] - min_s P[Y_hat=1|A=s,Y=1]
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    for combination in all_combinations:
        # Create mask for current subgroup with Y=1
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 1)

        # Calculate P[Y_hat=1|A=s,Y=1]
        if np.sum(mask) > 0:
            prob = np.sum(y_pred[mask] == 1) / np.sum(mask)
            probabilities.append(prob)

    if len(probabilities) == 0:
        return 0.0

    return max(probabilities) - min(probabilities)


def calculate_ped(y_true, y_pred, sensitive_data, sensitive_indices, value_ranges):
    """
    Calculate Predictive Equality Difference (PED)
    PED = max_s P[Y_hat=1|A=s,Y=0] - min_s P[Y_hat=1|A=s,Y=0]
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    for combination in all_combinations:
        # Create mask for current subgroup with Y=0
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 0)

        # Calculate P[Y_hat=1|A=s,Y=0]
        if np.sum(mask) > 0:
            prob = np.sum(y_pred[mask] == 1) / np.sum(mask)
            probabilities.append(prob)

    if len(probabilities) == 0:
        return 0.0

    return max(probabilities) - min(probabilities)


def calculate_aod(y_true, y_pred, sensitive_data, sensitive_indices, value_ranges):
    """
    Calculate Average Odds Difference (AOD)
    AOD = 0.5 * [max_s(P[Y_hat=1|A=s,Y=0] + P[Y_hat=1|A=s,Y=1]) -
                 min_s(P[Y_hat=1|A=s,Y=0] + P[Y_hat=1|A=s,Y=1])]
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    sums = []
    for combination in all_combinations:
        # Create base mask for current subgroup
        base_mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            base_mask &= (sensitive_data[:, attr_idx] == combination[i])

        # Calculate P[Y_hat=1|A=s,Y=0]
        mask_y0 = base_mask & (y_true.flatten() == 0)
        if np.sum(mask_y0) > 0:
            prob_y0 = np.sum(y_pred[mask_y0] == 1) / np.sum(mask_y0)
        else:
            prob_y0 = 0.0

        # Calculate P[Y_hat=1|A=s,Y=1]
        mask_y1 = base_mask & (y_true.flatten() == 1)
        if np.sum(mask_y1) > 0:
            prob_y1 = np.sum(y_pred[mask_y1] == 1) / np.sum(mask_y1)
        else:
            prob_y1 = 0.0

        sums.append(prob_y0 + prob_y1)

    if len(sums) == 0:
        return 0.0

    return 0.5 * (max(sums) - min(sums))


def generate_similar_samples(x, sensitive_indices, value_ranges):
    """
    Generate all similar samples by varying only the sensitive attributes.

    Args:
        x: Single sample (1D array)
        sensitive_indices: Indices of sensitive attributes
        value_ranges: Value ranges for each feature

    Returns:
        2D array of similar samples
    """
    if sensitive_indices == []:
        return x.reshape(1, -1)

    # Generate all combinations of sensitive attribute values
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    # Create similar samples
    similar_samples = []
    for combination in all_combinations:
        x_new = x.copy()
        for i, attr_idx in enumerate(sensitive_indices):
            x_new[attr_idx] = combination[i]
        similar_samples.append(x_new)

    return np.array(similar_samples)


def is_individually_discriminated(x, model_predict_func, sensitive_indices, value_ranges):
    """
    Check if a sample is individually discriminated.

    Args:
        x: Single sample (1D array)
        model_predict_func: Function that takes array of samples and returns predictions
        sensitive_indices: Indices of sensitive attributes
        value_ranges: Value ranges for each feature

    Returns:
        bool: True if the sample is discriminated, False otherwise
    """
    # Generate similar samples
    similar_samples = generate_similar_samples(x, sensitive_indices, value_ranges)

    # Get predictions for all similar samples
    predictions = model_predict_func(similar_samples)

    # Check if all predictions are the same
    # If not all predictions are the same, then there's discrimination
    return not (np.all(predictions == predictions[0]))


def calculate_individual_discrimination_rate(test_data, model_predict_func, sensitive_indices, value_ranges):
    """
    Calculate the individual discrimination rate for causal fairness.

    Args:
        test_data: 2D array of test samples
        model_predict_func: Function that takes array of samples and returns predictions
        sensitive_indices: Indices of sensitive attributes
        value_ranges: Value ranges for each feature

    Returns:
        float: Proportion of discriminated samples
    """
    n_discriminated = 0
    n_samples = len(test_data)

    for i in range(n_samples):
        if is_individually_discriminated(test_data[i], model_predict_func, sensitive_indices, value_ranges):
            n_discriminated += 1

    return n_discriminated / n_samples


def generate_global_similar_samples_advanced_sampling(x, value_ranges, delta=1, n_samples=None, strategy='random',
                                                      seed=None, sensitive_indices=None, non_sensitive_threshold=0.2):
    """
    Generate similar samples using different sampling strategies.

    Args:
        x: Single sample (1D array)
        value_ranges: Value ranges for each feature
        delta: Maximum allowed difference for each feature (for non-sensitive attributes)
        n_samples: Number of samples to generate
        strategy: Sampling strategy ('per_feature', 'random', 'systematic')
        seed: Random seed for reproducibility
        sensitive_indices: Indices of sensitive attributes (for strategy='random')
        non_sensitive_threshold: Threshold percentage for non-sensitive attribute perturbation (default 0.2)

    Returns:
        2D array of similar samples
    """
    if seed is not None:
        np.random.seed(seed)

    n_features = len(x)
    if n_samples is None:
        n_samples = n_features

    similar_samples = []
    similar_samples.append(x)

    if strategy == 'random':
        # Calculate unique value count for non-sensitive attributes based on value_ranges
        non_sensitive_perturbation_allowed = {}
        if sensitive_indices is not None:
            for i in range(n_features):
                if i not in sensitive_indices:
                    # Calculate unique value count from value_ranges
                    min_val, max_val = value_ranges[i]
                    unique_count = max_val - min_val + 1  # Number of unique values in integer range
                    # Calculate allowed perturbation magnitude
                    allowed_perturbation = non_sensitive_threshold * unique_count
                    # Allow perturbation if magnitude >= 1, otherwise not allowed
                    non_sensitive_perturbation_allowed[i] = allowed_perturbation >= 1.0

        for _ in range(n_samples):
            x_new = x.copy()

            for i in range(n_features):
                min_val, max_val = value_ranges[i]

                if sensitive_indices is not None and i in sensitive_indices:
                    # Sensitive attributes: randomly select any integer within valid range
                    possible_values = list(range(min_val, max_val + 1))
                    x_new[i] = np.random.choice(possible_values)

                elif sensitive_indices is not None:
                    # Non-sensitive attributes: determine whether perturbation is allowed based on threshold
                    if non_sensitive_perturbation_allowed.get(i, False):
                        # Perturbation allowed: randomly select from {-1, 0, 1}
                        change = np.random.choice([-1, 0, 1])
                        new_val = x[i] + change
                        x_new[i] = np.clip(new_val, min_val, max_val)
                    # If perturbation not allowed, keep original value (x_new[i] = x[i], already done by copy)

                else:
                    # Original logic: randomly select from {-delta, ..., +delta}
                    change = np.random.choice(range(-delta, delta + 1))
                    new_val = x[i] + change
                    x_new[i] = np.clip(new_val, min_val, max_val)

            similar_samples.append(x_new)

    elif strategy == 'per_feature':
        for i in range(min(n_samples, n_features)):
            x_new = x.copy()
            feature_idx = i % n_features
            min_val, max_val = value_ranges[feature_idx]

            possible_changes = list(range(-delta, delta + 1))
            if 0 in possible_changes and len(possible_changes) > 1:
                possible_changes.remove(0)
            if possible_changes:
                change = np.random.choice(possible_changes)
                new_val = x[feature_idx] + change
                x_new[feature_idx] = np.clip(new_val, min_val, max_val)

            similar_samples.append(x_new)

    elif strategy == 'systematic':
        for i in range(min(n_samples, n_features)):
            x_new = x.copy()
            feature_idx = i % n_features
            min_val, max_val = value_ranges[feature_idx]

            # Systematic selection of changes: alternating positive and negative
            change = delta if i % 2 == 0 else -delta
            new_val = x[feature_idx] + change
            x_new[feature_idx] = np.clip(new_val, min_val, max_val)

            similar_samples.append(x_new)

    return np.array(similar_samples)


def calculate_global_individual_fairness_sampling(test_data, model_predict_func, value_ranges, delta=1,
                                         n_samples_per_instance=None, strategy='random', seed=None,
                                         sensitive_indices=None, non_sensitive_threshold=0.2):
    """
    Calculate global individual fairness.

    Args:
        test_data: Test dataset (2D array)
        model_predict_func: Function to get model predictions
        value_ranges: Value ranges for each feature
        delta: Maximum allowed difference for each feature (for non-sensitive attributes)
        n_samples_per_instance: Number of samples to generate per instance
        strategy: Sampling strategy ('per_feature', 'random', 'systematic')
        seed: Random seed for reproducibility
        sensitive_indices: Indices of sensitive attributes
        non_sensitive_threshold: Threshold percentage for non-sensitive attribute perturbation (default 0.2)

    Returns:
        float: Proportion of discriminated instances (global individual fairness score)
    """
    n_discriminated = 0
    n_samples = len(test_data)

    for i in range(n_samples):
        # Generate similar samples with sampling
        similar_samples = generate_global_similar_samples_advanced_sampling(
            test_data[i], value_ranges, delta, n_samples_per_instance, strategy, seed,
            sensitive_indices=sensitive_indices,
            non_sensitive_threshold=non_sensitive_threshold
        )

        # Get predictions
        predictions = model_predict_func(similar_samples)

        # Check if discrimination exists
        if not np.all(predictions == predictions[0]):
            n_discriminated += 1

    return n_discriminated / n_samples


# Original exhaustive methods (kept for backward compatibility)
def generate_global_similar_samples(x, value_ranges, delta=1):
    """
    Generate all similar samples where each feature differs by at most delta.
    WARNING: This has exponential complexity and should only be used for small datasets.

    Args:
        x: Single sample (1D array)
        value_ranges: Value ranges for each feature [[min, max], ...]
        delta: Maximum allowed difference for each feature

    Returns:
        2D array of similar samples (including the original sample)
    """
    n_features = len(x)

    # For each feature, determine valid values within delta range
    feature_domains = []
    for i in range(n_features):
        min_val, max_val = value_ranges[i]
        # Values within delta of x[i] and within valid range
        valid_values = []
        for val in range(min_val, max_val + 1):
            if abs(val - x[i]) <= delta:
                valid_values.append(val)
        feature_domains.append(valid_values)

    # Generate all combinations
    all_combinations = list(itertools.product(*feature_domains))

    # Convert to numpy array
    similar_samples = np.array(all_combinations)

    return similar_samples


def calculate_global_individual_fairness(test_data, model_predict_func, value_ranges, delta=1):
    """
    Calculate Global Individual Fairness (exhaustive method).
    WARNING: This has exponential complexity and should only be used for small datasets.

    For each sample x, find all similar samples x' where |x_i - x'_i| <= delta for all features i.
    If any similar sample gets a different prediction, the original sample is considered discriminated.

    Args:
        test_data: 2D array of test samples
        model_predict_func: Function that takes array of samples and returns predictions
        value_ranges: Value ranges for each feature [[min, max], ...]
        delta: Similarity threshold (default=1)

    Returns:
        float: Proportion of discriminated samples (lower is better, 0 is perfect fairness)
    """
    n_discriminated = 0
    n_samples = len(test_data)

    for i in range(n_samples):
        x = test_data[i]

        # Generate all possible similar samples
        similar_samples = generate_global_similar_samples(x, value_ranges, delta)

        # Get predictions for all similar samples (including original)
        predictions = model_predict_func(similar_samples)

        # Check if all predictions are the same
        if not np.all(predictions == predictions[0]):
            n_discriminated += 1

    return n_discriminated / n_samples


def measure_final_score(test_data, y_true, y_pred, sensitive_indices, value_ranges,
                        y_pred_proba=None, model_predict_func=None, awareness=None, delta=1,
                        global_fairness_samples=None, use_sampling=True, sampling_strategy='random', non_sensitive_threshold=0.2, seed=None):
    """
    Compute common metrics of model utilities and fairness for binary classification.

    Args:
        test_data: 2D numpy array of test data features
        y_true: 1D or 2D numpy array of true binary labels
        y_pred: 1D or 2D numpy array of predicted binary labels
        sensitive_indices: array of indices indicating which columns are sensitive attributes
        value_ranges: 2D numpy array where each row contains [min_value, max_value] for each feature
        y_pred_proba: 1D or 2D numpy array of predicted probabilities for positive class (optional)
        model_predict_func: Function that takes array of samples and returns binary predictions (optional)
        delta: Similarity threshold for global individual fairness (default=1)
        global_fairness_samples: Number of samples for global fairness (default: number of features)
        use_sampling: Whether to use sampling approach for global fairness (default=True)
        sampling_strategy: Strategy for sampling ('random', 'per_feature', 'systematic')
        non_sensitive_threshold: Threshold percentage for non-sensitive attribute perturbation (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        dict: Dictionary containing model performance and fairness metrics
    """
    # Ensure labels are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate model performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    fpr = calculate_fpr(y_true, y_pred)
    f1score_macro = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)

    model_metrics = {}

    model_metrics["utilities"] = {
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "FPR": fpr,
        "f1_score": f1score_macro,
        "MCC": mcc
    }

    # Calculate ROC-AUC if probability predictions are provided
    if y_pred_proba is not None:
        y_pred_proba = y_pred_proba.flatten()
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        model_metrics["utilities"]["roc_auc"] = roc_auc

    # Calculate fairness metrics
    model_metrics["fairness"] = {}
    if len(sensitive_indices) > 0:
        spd = calculate_spd(y_pred, test_data, sensitive_indices, value_ranges)
        eod = calculate_eod(y_true, y_pred, test_data, sensitive_indices, value_ranges)
        ped = calculate_ped(y_true, y_pred, test_data, sensitive_indices, value_ranges)
        aod = calculate_aod(y_true, y_pred, test_data, sensitive_indices, value_ranges)

        model_metrics["fairness"]["SPD"] = spd
        model_metrics["fairness"]["EOD"] = eod
        model_metrics["fairness"]["PED"] = ped
        model_metrics["fairness"]["AOD"] = aod

        # Calculate individual discrimination rate (causal fairness)
        if model_predict_func is not None:
            if awareness == "without_SA":
                model_metrics["fairness"]["CFVR"] = 0
            else:
                model_metrics["fairness"]["CFVR"] = calculate_individual_discrimination_rate(test_data, model_predict_func,
                                                               sensitive_indices, value_ranges)

    # Calculate Global Individual Fairness
    if model_predict_func is not None:
        if awareness == "without_SA":
            test_data = np.delete(test_data, sensitive_indices, axis=1)
            value_ranges = np.delete(value_ranges, sensitive_indices, axis=0)
            sensitive_indices = []
        if use_sampling:
            gif = calculate_global_individual_fairness_sampling(
                test_data, model_predict_func, value_ranges, delta,
                global_fairness_samples, sampling_strategy, seed, sensitive_indices, non_sensitive_threshold)
            model_metrics["fairness"]["GIFVR"] = gif
        else:
            # Use original exhaustive method (only for small datasets)
            gif = calculate_global_individual_fairness(test_data, model_predict_func,
                                                       value_ranges, delta)
            model_metrics["fairness"]["GIFVR_exhaustive"] = gif

    return model_metrics