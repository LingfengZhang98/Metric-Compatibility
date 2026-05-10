import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from itertools import combinations
import numpy as np
import joblib
import json
import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

from tools.utils import set_seed, makedirs, NumpyEncoder
from tools.config import considered_sensitive_attributes, RANDOM_SEED_LIST, DATASETS, list_group_fairness, list_individual_fairness, list_utilities, COLORS_PAIR
from tools.evaluation import measure_final_score
from tools.models import LogisticRegressionModel
from tools.decomposition import calculate_harsanyi_interactions, decompose_metrics
from exp.calculate_contribution_and_compatibility import calculate_compatibility


METHOD_NAME = "hifi"
ETA_LIST = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]


def get_masked_inputs(inputs, list_activated, mean_values):
    """
    Mask the variables of {inputs} not in the {list_activated} with {mean_values}.
    """
    masked_inputs = inputs.clone()
    mask = torch.ones(inputs.size(1), dtype=torch.bool)
    mask[list_activated] = False
    masked_inputs[:, mask] = mean_values[mask]
    return masked_inputs


def get_all_nonempty_subsets(input_list):
    subsets = []
    for r in range(len(input_list) + 1):
        subsets.extend(combinations(input_list, r))
    return [list(subset) for subset in subsets if list(subset)]


def get_all_subsets(input_list):
    subsets = []
    for r in range(len(input_list) + 1):
        subsets.extend(combinations(input_list, r))
    return [list(subset) for subset in subsets]


def custom_loss(classifier_name, model, inputs, outputs, labels, indices_sensitive_attributes, eta):
    # compute the original classification loss
    classifier_name = classifier_name.upper()
    if classifier_name in ["LR"]:
        classification_loss = nn.BCELoss()(outputs, labels)
    else:
        raise NotImplementedError(f"HIFI: {classifier_name} has not been implemented.")

    # compute the additional loss terms from HIFI
    hi_loss = 0
    if eta != 0:
        mean_values = torch.mean(inputs, dim=0)
        sensitive_coalitions = get_all_nonempty_subsets(indices_sensitive_attributes)
        for sensitive_coalition in sensitive_coalitions:
            interactions = torch.zeros_like(outputs)
            subsets = get_all_subsets(sensitive_coalition)
            for subset in subsets:
                masked_inputs = get_masked_inputs(inputs, subset, mean_values)
                outputs_on_masked_inputs = model(masked_inputs)
                interactions = interactions + ((-1)**(len(sensitive_coalition)-len(subset))) * outputs_on_masked_inputs
            hi_loss += torch.mean(torch.abs(interactions))

    return classification_loss + eta * hi_loss


def train_with_hifi(dataset_name, classifier_name="lr", eta=0.75, seed=42):
    method_name = METHOD_NAME
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    
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
    
    # Setup model save path
    model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                classifier_name, f"seed_{seed}")
    model_save_root = osp.join(model_save_dir, f"{classifier_name}_eta={eta}")
    makedirs(model_save_dir)
    
    print(f"[{dataset_name}|{classifier_name}|seed={seed}|eta={eta}] Training started...")
    
    classifier_name = classifier_name.upper()
    if classifier_name in ["LR"]:
        clf = LogisticRegressionModel(X_train_scaled.shape[1])
    else:
        raise NotImplementedError(f"HIFI: {classifier_name} has not been implemented.")
    clf.to(device)
    
    X = torch.from_numpy(X_train_scaled).float()
    y = torch.from_numpy(y_train).float()
    batch_size = 256 if X_train_scaled.shape[0] > 10000 else 64
    data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    
    optimizer = optim.NAdam(clf.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    num_epochs = 100
    tolerance = 1e-5
    previous_loss = float('inf')
    early_stop = False
    for epoch in range(num_epochs):
        clf.train()
        epoch_loss = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = custom_loss(classifier_name, clf, inputs, outputs, labels, sensitive_indices, eta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')
        if abs(previous_loss - avg_epoch_loss) < tolerance:
            print(f'Training converged at epoch {epoch + 1}')
            early_stop = True
            break
        previous_loss = avg_epoch_loss
    if not early_stop:
        print('Reached the maximum number of epochs without convergence.')
    
    clf.eval()
    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test_scaled).float().to(device)
        y_pred = (clf(X_test_torch) > 0.5).int().cpu().numpy()
        y_pred_proba = clf(X_test_torch).cpu().numpy()
        
        def predict_func(inputs):
            X_scaled = scaler.transform(inputs)
            X_scaled = torch.from_numpy(X_scaled).float().to(device)
            return (clf(X_scaled) > 0.5).int().cpu().numpy()
        
        eval_result = measure_final_score(data_test[:, :-1], y_test, y_pred, 
                                         sensitive_indices, constraints, 
                                         y_pred_proba, predict_func, seed=seed)
        
        # Save metrics
        with open(model_save_root + "_metrics.json", "w") as f:
            json.dump(eval_result, f, indent=1, cls=NumpyEncoder)
            
        # Save model
        torch.save(clf.state_dict(), model_save_root + ".pth")
        
        print(f"[{dataset_name}|{classifier_name}|seed={seed}|eta={eta}] ✓ Completed")


def classify_metric_pair(metric_pair):
    """Classify metric pair into categories based on metric types"""
    metrics = metric_pair.split('-')
    if len(metrics) != 2:
        return None
    
    metric1, metric2 = metrics
    
    def get_metric_type(metric):
        if metric in list_group_fairness:
            return 'Group Fairness'
        elif metric in list_individual_fairness:
            return 'Individual Fairness'
        elif metric in list_utilities:
            return 'Utility'
        return None
    
    type1 = get_metric_type(metric1)
    type2 = get_metric_type(metric2)
    
    if type1 is None or type2 is None:
        return None
    
    if type1 == type2:
        return f'{type1} vs {type2}'
    else:
        types = sorted([type1, type2])
        return f'{types[0]} vs {types[1]}'


def normalize_metric_pair(metric_pair):
    """Normalize metric pair to ensure consistent ordering"""
    metrics = sorted(metric_pair.split('-'))
    return '-'.join(metrics)


# Monotonic decreasing function with adjustable sharpness
def monotonic_decrease(x, x_mid, y_start, y_end, sharpness):
    """Monotonic decreasing sigmoid-like function"""
    # Normalize x to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    # Sigmoid with adjustable sharpness
    sigmoid = 1 / (1 + np.exp(sharpness * (x_norm - x_mid)))
    # Scale to y range
    return y_start + (y_end - y_start) * (1 - sigmoid)


# Find optimal breakpoint for piecewise linear fit
def find_optimal_piecewise_linear(x_sorted, y_sorted):
    """
    Find the optimal breakpoint by trying each data point as a potential breakpoint
    and selecting the one with minimum fitting error (MSE).
    """
    n = len(x_sorted)
    if n < 3:
        return None, None, None
    
    best_mse = float('inf')
    best_break_idx = None
    best_params = None
    
    # Try each interior point as a potential breakpoint (exclude first and last)
    for break_idx in range(1, n - 1):
        # Split data at this breakpoint
        x1 = x_sorted[:break_idx + 1]
        y1 = y_sorted[:break_idx + 1]
        x2 = x_sorted[break_idx:]
        y2 = y_sorted[break_idx:]
        
        # Fit two linear segments
        # Segment 1: from start to breakpoint
        if len(x1) >= 2:
            slope1 = (y1[-1] - y1[0]) / (x1[-1] - x1[0]) if x1[-1] != x1[0] else 0
            intercept1 = y1[0] - slope1 * x1[0]
        else:
            continue
        
        # Segment 2: from breakpoint to end
        if len(x2) >= 2:
            slope2 = (y2[-1] - y2[0]) / (x2[-1] - x2[0]) if x2[-1] != x2[0] else 0
            intercept2 = y2[0] - slope2 * x2[0]
        else:
            continue
        
        # Calculate predictions
        y_pred = np.zeros_like(y_sorted)
        y_pred[:break_idx + 1] = slope1 * x_sorted[:break_idx + 1] + intercept1
        y_pred[break_idx:] = slope2 * x_sorted[break_idx:] + intercept2
        
        # Calculate MSE
        mse = np.mean((y_sorted - y_pred) ** 2)
        
        # Update best if this is better
        if mse < best_mse:
            best_mse = mse
            best_break_idx = break_idx
            best_params = {
                'break_x': x_sorted[break_idx],
                'break_y': y_sorted[break_idx],
                'slope1': slope1,
                'intercept1': intercept1,
                'slope2': slope2,
                'intercept2': intercept2
            }
    
    return best_break_idx, best_params, best_mse


# Piecewise linear function using precomputed parameters
def piecewise_linear_predict(x, break_x, slope1, intercept1, slope2, intercept2):
    """Predict using piecewise linear model"""
    result = np.zeros_like(x)
    mask1 = x <= break_x
    result[mask1] = slope1 * x[mask1] + intercept1
    result[~mask1] = slope2 * x[~mask1] + intercept2
    return result


# Helper function to plot trade-off with adaptive curve fitting
def plot_tradeoff(ax, x_data, y_data, x_label, y_label, title, subplot_idx):
    # Remove any None or NaN values
    valid_indices = [i for i in range(len(x_data)) if x_data[i] is not None and y_data[i] is not None]
    x_valid = np.array([x_data[i] for i in valid_indices])
    y_valid = np.array([y_data[i] for i in valid_indices])
    eta_valid = [ETA_LIST[i] for i in valid_indices]
    
    # Create color map based on eta values (log scale for better visualization)
    eta_log = np.log10(np.array(eta_valid) + 1)  # +1 to handle eta=0
    eta_normalized = (eta_log - eta_log.min()) / (eta_log.max() - eta_log.min())
    
    # Plot scatter points with varying colors
    scatter = ax.scatter(x_valid, y_valid, s=100, alpha=0.8, c=eta_normalized, 
                        cmap='viridis', edgecolors='black', linewidth=1.0, 
                        zorder=3, vmin=0, vmax=1)
    
    # Fit curve based on subplot
    if len(x_valid) > 3:
        try:
            # Sort by x values
            sort_idx = np.argsort(x_valid)
            x_sorted = x_valid[sort_idx]
            y_sorted = y_valid[sort_idx]
            
            # Generate smooth curve
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            
            if subplot_idx == 0:
                # First subplot: use parabolic fit
                z = np.polyfit(x_sorted, y_sorted, 2)
                p = np.poly1d(z)
                y_smooth = p(x_smooth)
                
                # Store fitting function
                fitting_functions['subplot_1_accuracy_vs_AOD'] = {
                    'type': 'polynomial',
                    'degree': 2,
                    'coefficients': z.tolist(),
                    'formula': f'y = {z[0]:.6f}*x^2 + {z[1]:.6f}*x + {z[2]:.6f}',
                    'x_range': [float(x_sorted.min()), float(x_sorted.max())],
                    'y_range': [float(y_sorted.min()), float(y_sorted.max())]
                }
                
            elif subplot_idx == 1:
                # Second subplot: use monotonic decreasing function (gentle then sharp)
                try:
                    # Normalize for fitting
                    x_norm = (x_sorted - x_sorted.min()) / (x_sorted.max() - x_sorted.min())
                    
                    # Initial parameters
                    x_mid_init = 0.3  # Transition point
                    y_start_init = y_sorted.max()
                    y_end_init = y_sorted.min()
                    sharpness_init = 10  # Controls how sharp the drop is
                    
                    popt, _ = curve_fit(
                        lambda x, x_mid, y_start, y_end, sharpness: monotonic_decrease(
                            x, x_mid, y_start, y_end, sharpness
                        ),
                        x_norm, y_sorted,
                        p0=[x_mid_init, y_start_init, y_end_init, sharpness_init],
                        bounds=([0, y_sorted.min(), y_sorted.min(), 1], 
                               [1, y_sorted.max(), y_sorted.max(), 50]),
                        maxfev=10000
                    )
                    
                    # Generate smooth curve
                    x_smooth_norm = (x_smooth - x_sorted.min()) / (x_sorted.max() - x_sorted.min())
                    y_smooth = monotonic_decrease(x_smooth_norm, *popt)
                    
                    # Store fitting function
                    fitting_functions['subplot_2_accuracy_vs_CFVR'] = {
                        'type': 'monotonic_sigmoid',
                        'parameters': {
                            'x_mid': float(popt[0]),
                            'y_start': float(popt[1]),
                            'y_end': float(popt[2]),
                            'sharpness': float(popt[3])
                        },
                        'formula': 'y = y_start + (y_end - y_start) * (1 - 1/(1 + exp(sharpness * (x_norm - x_mid))))',
                        'note': 'x_norm = (x - x_min) / (x_max - x_min)',
                        'x_range': [float(x_sorted.min()), float(x_sorted.max())],
                        'y_range': [float(y_sorted.min()), float(y_sorted.max())]
                    }
                    
                except Exception as e:
                    print(f"Monotonic fit failed for subplot {subplot_idx}: {e}, using spline")
                    from scipy.interpolate import UnivariateSpline
                    spl = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*0.1, k=3)
                    y_smooth = spl(x_smooth)
                    
                    fitting_functions['subplot_2_accuracy_vs_CFVR'] = {
                        'type': 'spline',
                        'note': 'Fallback to spline interpolation due to fitting failure'
                    }
                    
            else:
                # Third subplot: use optimal piecewise linear
                best_break_idx, best_params, best_mse = find_optimal_piecewise_linear(x_sorted, y_sorted)
                
                if best_params is not None:
                    # Generate smooth curve using optimal parameters
                    y_smooth = piecewise_linear_predict(
                        x_smooth,
                        best_params['break_x'],
                        best_params['slope1'],
                        best_params['intercept1'],
                        best_params['slope2'],
                        best_params['intercept2']
                    )
                    
                    print(f"\nSubplot 3 optimal piecewise linear fit:")
                    print(f"  Break point index: {best_break_idx} (x={best_params['break_x']:.4f}, y={best_params['break_y']:.4f})")
                    print(f"  Segment 1: slope={best_params['slope1']:.4f}, intercept={best_params['intercept1']:.4f}")
                    print(f"  Segment 2: slope={best_params['slope2']:.4f}, intercept={best_params['intercept2']:.4f}")
                    print(f"  MSE: {best_mse:.6f}")
                    
                    # Store fitting function
                    fitting_functions['subplot_3_AOD_vs_CFVR'] = {
                        'type': 'piecewise_linear',
                        'breakpoint': {
                            'x': float(best_params['break_x']),
                            'y': float(best_params['break_y']),
                            'index': int(best_break_idx)
                        },
                        'segment_1': {
                            'slope': float(best_params['slope1']),
                            'intercept': float(best_params['intercept1']),
                            'formula': f"y = {best_params['slope1']:.6f}*x + {best_params['intercept1']:.6f}",
                            'x_range': [float(x_sorted.min()), float(best_params['break_x'])]
                        },
                        'segment_2': {
                            'slope': float(best_params['slope2']),
                            'intercept': float(best_params['intercept2']),
                            'formula': f"y = {best_params['slope2']:.6f}*x + {best_params['intercept2']:.6f}",
                            'x_range': [float(best_params['break_x']), float(x_sorted.max())]
                        },
                        'mse': float(best_mse),
                        'x_range': [float(x_sorted.min()), float(x_sorted.max())],
                        'y_range': [float(y_sorted.min()), float(y_sorted.max())]
                    }
                    
                    # Mark the breakpoint on the plot (without legend)
                    # ax.plot(best_params['break_x'], best_params['break_y'], 
                    #        'r*', markersize=15, zorder=5)
                else:
                    print(f"Could not find optimal breakpoint for subplot {subplot_idx}, using polynomial")
                    z = np.polyfit(x_sorted, y_sorted, min(3, len(x_sorted)-1))
                    p = np.poly1d(z)
                    y_smooth = p(x_smooth)
                    
                    fitting_functions['subplot_3_AOD_vs_CFVR'] = {
                        'type': 'polynomial_fallback',
                        'degree': min(3, len(x_sorted)-1),
                        'coefficients': z.tolist(),
                        'note': 'Fallback to polynomial due to piecewise fitting failure'
                    }
            
            ax.plot(x_smooth, y_smooth, '-', color='#A23B72', linewidth=3, 
                   alpha=0.5, zorder=2, linestyle='--')
        except Exception as e:
            print(f"Curve fitting failed for subplot {subplot_idx}: {e}")
    
    # Add eta labels only for 0.1 and 1000
    label_etas = [0.1, 1000]
    for idx, eta_val in enumerate(eta_valid):
        if eta_val in label_etas:
            if eta_val == 0.1:
                label = '0.1'
            elif eta_val == 1000:
                label = '10³'
            
            ax.annotate(f'η={label}', 
                    xy=(x_valid[idx], y_valid[idx]),
                    xytext=(7, -15), textcoords='offset points',
                    fontsize=11, alpha=0.85, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='gray', alpha=0.85, linewidth=0.6),
                    zorder=4)

    
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(r'$\eta$ (log scale)', fontsize=11, rotation=270, labelpad=18)
    cbar.ax.tick_params(labelsize=10)
    # Set colorbar ticks
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = []
    for pos in tick_positions:
        eta_log_val = eta_log.min() + pos * (eta_log.max() - eta_log.min())
        eta_val = 10**eta_log_val - 1
        if eta_val < 0.01:
            tick_labels.append('0')
        elif eta_val < 1:
            tick_labels.append(f'{eta_val:.2f}')
        elif eta_val < 1000:
            tick_labels.append(f'{int(eta_val)}')
        else:
            tick_labels.append(f'{int(eta_val):.0e}')
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    # Add arrows to indicate optimal direction
    if subplot_idx in [0, 1]:
        # Subplots 1 and 2: arrow pointing to top-left (↖)
        ax.annotate('', xy=(0.15, 0.85), xytext=(0.25, 0.75),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->,head_width=0.8,head_length=0.8', 
                                lw=6, color='red', alpha=0.4),
                    zorder=10)
    elif subplot_idx == 2:
        # Subplot 3: arrow pointing to bottom-left (↙)
        ax.annotate('', xy=(0.15, 0.15), xytext=(0.25, 0.25),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->,head_width=0.8,head_length=0.8', 
                                lw=6, color='red', alpha=0.4),
                    zorder=10)


if __name__ == '__main__':
    n_process = 2  # number of parallel processes
    with mp.Pool(processes=n_process) as pool:
        for dataset in DATASETS:
            for seed in RANDOM_SEED_LIST:
                for eta in ETA_LIST:
                    pool.apply_async(train_with_hifi, args=(dataset, "lr", eta, seed))

        pool.close()
        pool.join()
        
    results_dir = osp.join(script_dir, "results")
    makedirs(results_dir)
    
    # Store all metric values for each eta
    # Structure: eta_metrics[eta][category][metric] = [value1, value2, ...]
    eta_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Iterate through all combinations and read JSON files
    for dataset in DATASETS:
        for seed in RANDOM_SEED_LIST:
            for eta in ETA_LIST:
                metric_save_root = osp.join(
                    script_dir, "../models", dataset, METHOD_NAME, 
                    "lr", f"seed_{seed}", f"lr_eta={eta}_metrics.json"
                )
                
                # Check if file exists
                if not osp.exists(metric_save_root):
                    print(f"Warning: File not found - {metric_save_root}")
                    continue
                
                # Read JSON file
                with open(metric_save_root, 'r') as f:
                    data = json.load(f)
                
                # Store data categorized by eta
                for category in data:  # 'utilities', 'fairness'
                    for metric, value in data[category].items():
                        eta_metrics[eta][category][metric].append(value)

    # Build result structure with eta_list at the beginning
    result = {"eta_list": ETA_LIST, "utilities": {}, "fairness": {}}

    # Get all metric names from the first valid eta
    for eta in ETA_LIST:
        if eta in eta_metrics:
            for category in eta_metrics[eta]:
                for metric in eta_metrics[eta][category]:
                    if metric not in result[category]:
                        result[category][metric] = []

    # Calculate average for each eta
    for eta in ETA_LIST:
        if eta in eta_metrics:
            for category in ["utilities", "fairness"]:
                for metric in result[category]:
                    values = eta_metrics[eta][category][metric]
                    avg_value = np.mean(values) if values else 0.0
                    result[category][metric].append(avg_value)
        else:
            # Fill with 0 if no data for this eta
            for category in ["utilities", "fairness"]:
                for metric in result[category]:
                    result[category][metric].append(0.0)

    # Save results
    output_path = osp.join(results_dir, f"{METHOD_NAME}_averaged_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {output_path}")
    print(f"Processed {len(DATASETS)} datasets × {len(RANDOM_SEED_LIST)} seeds × {len(ETA_LIST)} etas")
    
    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        for dataset in DATASETS:
            for seed in RANDOM_SEED_LIST:
                for eta in ETA_LIST:
                    pool.apply_async(
                        calculate_harsanyi_interactions,
                        args=(dataset, METHOD_NAME, "lr", seed),
                        kwds={'hifi_eta': eta}
                    )
        pool.close()
        pool.join()

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        for dataset in DATASETS:
            for seed in RANDOM_SEED_LIST:
                for eta in ETA_LIST:
                    pool.apply_async(
                        decompose_metrics,
                        args=(dataset, METHOD_NAME, "lr", seed),
                        kwds={'hifi_eta': eta}
                    )
        pool.close()
        pool.join()

    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        for dataset in DATASETS:
            for seed in RANDOM_SEED_LIST:
                for eta in ETA_LIST:
                    pool.apply_async(
                        calculate_compatibility,
                        args=(dataset, METHOD_NAME, "lr", seed),
                        kwds={'hifi_eta': eta}
                    )
        pool.close()
        pool.join()

    # Load averaged metrics
    averaged_metrics_path = osp.join(results_dir, f"{METHOD_NAME}_averaged_metrics.json")
    with open(averaged_metrics_path, 'r') as f:
        averaged_metrics = json.load(f)

    # Step 1: Collect and average compatibility data
    print("Collecting compatibility data...")
    compatibility_data = defaultdict(lambda: defaultdict(list))

    for dataset in DATASETS:
        for seed in RANDOM_SEED_LIST:
            for eta in ETA_LIST:
                compatibility_path = osp.join(
                    script_dir, "../models", dataset, METHOD_NAME, 
                    "lr", f"seed_{seed}", f"interactions_eta={eta}",
                    "compatibility", "compatibility.json"
                )
                
                if osp.exists(compatibility_path):
                    with open(compatibility_path, 'r') as f:
                        comp_data = json.load(f)
                        for pair, value in comp_data.items():
                            normalized_pair = normalize_metric_pair(pair)
                            compatibility_data[normalized_pair][eta].append(value)
                else:
                    print(f"Warning: File not found: {compatibility_path}")

    # Calculate averaged compatibility
    averaged_compatibility = {"eta_list": ETA_LIST}
    for pair in compatibility_data:
        averaged_compatibility[pair] = []
        for eta in ETA_LIST:
            if eta in compatibility_data[pair] and len(compatibility_data[pair][eta]) > 0:
                avg_value = np.mean(compatibility_data[pair][eta])
                averaged_compatibility[pair].append(avg_value)
            else:
                averaged_compatibility[pair].append(None)

    # Save averaged compatibility
    output_compatibility_path = osp.join(results_dir, f"{METHOD_NAME}_averaged_compatibility.json")
    with open(output_compatibility_path, 'w') as f:
        json.dump(averaged_compatibility, f, indent=2)
    print(f"Saved averaged compatibility to: {output_compatibility_path}")

    # Step 2: Classify and group compatibility by pair types
    pair_type_data = defaultdict(lambda: defaultdict(list))
    pair_type_members = defaultdict(list)  # Track which pairs belong to each type

    for pair, values in averaged_compatibility.items():
        if pair == "eta_list":
            continue
        
        pair_type = classify_metric_pair(pair)
        if pair_type is not None:
            pair_type_members[pair_type].append(pair)
            for i, eta in enumerate(ETA_LIST):
                if values[i] is not None:
                    pair_type_data[pair_type][eta].append(values[i])

    # Print classification results
    print("\n" + "="*60)
    print("METRIC PAIR CLASSIFICATION RESULTS")
    print("="*60)
    for pair_type in sorted(COLORS_PAIR.keys()):
        count = len(pair_type_members[pair_type])
        print(f"\n{pair_type}: {count} pairs")
        if count > 0:
            print(f"  Pairs: {', '.join(sorted(pair_type_members[pair_type]))}")
    print("="*60 + "\n")

    # Calculate average compatibility for each pair type
    pair_type_averaged = {}
    for pair_type in COLORS_PAIR.keys():
        pair_type_averaged[pair_type] = []
        for eta in ETA_LIST:
            if eta in pair_type_data[pair_type] and len(pair_type_data[pair_type][eta]) > 0:
                avg_value = np.mean(pair_type_data[pair_type][eta])
                pair_type_averaged[pair_type].append(avg_value)
            else:
                pair_type_averaged[pair_type].append(np.nan)

    # Step 3: Create the 4-subplot figure with custom layout
    fig = plt.figure(figsize=(24, 5))

    # Create custom grid: first 3 subplots normal, last one with space for legend on top
    gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Dictionary to store fitting functions for all subplots
    fitting_functions = {}

    # Subplot 1: accuracy vs AOD
    plot_tradeoff(
        axes[0],
        averaged_metrics['fairness']['AOD'],
        averaged_metrics['utilities']['accuracy'],
        'AOD',
        'Accuracy',
        'Trade-off between Accuracy and AOD',
        0
    )

    # Subplot 2: accuracy vs CFVR
    plot_tradeoff(
        axes[1],
        averaged_metrics['fairness']['CFVR'],
        averaged_metrics['utilities']['accuracy'],
        'CFVR',
        'Accuracy',
        'Trade-off between Accuracy and CFVR',
        1
    )

    # Subplot 3: AOD vs CFVR
    plot_tradeoff(
        axes[2],
        averaged_metrics['fairness']['CFVR'],
        averaged_metrics['fairness']['AOD'],
        'CFVR',
        'AOD',
        'Trade-off between AOD and CFVR',
        2
    )

    # Subplot 4: Inter-metric Compatibility (with compressed height for legend)
    ax4 = axes[3]
    # Adjust the position to leave space at top for legend
    pos = ax4.get_position()
    ax4.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.83])  # Compress to 83% height

    eta_indices = np.arange(len(ETA_LIST))

    for pair_type, color in COLORS_PAIR.items():
        if pair_type in pair_type_averaged:
            values = pair_type_averaged[pair_type]
            # Filter out NaN values for plotting
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax4.plot(eta_indices[valid_mask], np.array(values)[valid_mask], 
                        marker='o', linewidth=2.5, markersize=7, 
                        color=color, label=pair_type, alpha=0.8)

    ax4.set_xlabel(r'$\eta$', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Compatibility', fontsize=14, fontweight='bold')
    ax4.set_title('Inter-metric Compatibility', fontsize=14, fontweight='bold', pad=60)
    ax4.set_xticks(eta_indices)

    # Format x-axis labels as powers of 10
    eta_labels = []
    for eta in ETA_LIST:
        if eta == 0:
            eta_labels.append('0')
        else:
            power = int(np.log10(eta))
            eta_labels.append(f'10$^{{{power}}}$')

    ax4.set_xticklabels(eta_labels, rotation=45, ha='right', fontsize=11)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)

    # Place legend above the plot in the space we created
    legend = ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                        ncol=2, fontsize=9, framealpha=0.95, 
                        edgecolor='gray', fancybox=True, columnspacing=1.5,
                        handlelength=2.5, handletextpad=0.5)

    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    output_figure_path = osp.join(results_dir, f"{METHOD_NAME}_analysis.png")
    plt.savefig(output_figure_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_figure_path}")
    plt.show()

    # Save fitting functions to JSON file
    import json
    fitting_func_path = osp.join(results_dir, "hifi_tradeoff_fitting_func.json")
    with open(fitting_func_path, 'w') as f:
        json.dump(fitting_functions, f, indent=4)
    print(f"\nSaved fitting functions to: {fitting_func_path}")

    print("\nAnalysis completed!")