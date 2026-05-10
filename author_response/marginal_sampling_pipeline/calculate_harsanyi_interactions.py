"""
Marginal Sampling Distributional Masking for Harsanyi Interactions
Step 1: Calculate Harsanyi interactions using marginal sampling.

IMPROVED VERSION:
- K increased from 20 to 100
- Stratified sampling instead of pure random sampling for lower variance
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os.path as osp
import sys
import json
import time
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import joblib

script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")
sys.path.append(project_root)

from tools.config import (
    considered_sensitive_attributes, METRICS, DATASETS
)
from tools.utils import set_seed, makedirs, NumpyEncoder, cosine_similarity
from tools.models import load_classifier, model_standardize


def generate_masks(n_players):
    """Generate masks matrix consistent with AndHarsanyi encoding."""
    n_subsets = 2 ** n_players
    masks = np.zeros((n_subsets, n_players), dtype=bool)
    for i in range(n_subsets):
        for j in range(n_players):
            masks[i, j] = bool((i >> (n_players - 1 - j)) & 1)
    return masks


def mobius_inversion(rewards):
    """Fast Möbius inversion O(n * 2^n)."""
    interactions = rewards.copy()
    n = int(np.log2(rewards.shape[0]))
    for j in range(n):
        for i in range(2**n):
            if (i >> j) & 1:
                interactions[i] -= interactions[i ^ (1 << j)]
    return interactions


def compute_marginal_rewards(model, x, train_X, masks, K=100, rng=None):
    """
    Compute marginal rewards using stratified sampling.

    For each masked feature, divide train_X values into K strata,
    sample one value from each stratum. This reduces variance vs pure random.

    Parameters:
        model: callable, model(X) returns shape (m, 1) predictions
        x: shape (n_features,) single sample (raw features)
        train_X: shape (N_train, n_features) training set features (raw)
        masks: shape (2^n, n_features) bool array
        K: number of strata/samples per subset
        rng: numpy RandomState instance
    """
    if rng is None:
        rng = np.random.RandomState(0)

    n_subsets, n_features = masks.shape
    rewards = np.zeros((n_subsets, 1))

    # Pre-compute stratified sampling indices for each feature
    # For each feature j, sort train_X[:, j] and store sorted indices
    n_train = len(train_X)
    sorted_indices_per_feature = []
    for j in range(n_features):
        sorted_idx = np.argsort(train_X[:, j])
        sorted_indices_per_feature.append(sorted_idx)

    for i in range(n_subsets):
        batch = np.tile(x, (K, 1))  # (K, n_features)
        masked_indices = np.where(~masks[i])[0]

        for feat_idx in masked_indices:
            sorted_idx = sorted_indices_per_feature[feat_idx]
            # Divide sorted train samples into K strata
            strata_boundaries = np.linspace(0, n_train, K + 1, dtype=int)
            sampled_values = np.empty(K)

            for k in range(K):
                start = strata_boundaries[k]
                end = strata_boundaries[k + 1]
                if end > start:
                    chosen = rng.randint(start, end)
                else:
                    chosen = start
                sampled_values[k] = train_X[sorted_idx[chosen], feat_idx]

            # Shuffle to decorrelate strata
            rng.shuffle(sampled_values)
            batch[:, feat_idx] = sampled_values

        preds = model(batch)
        rewards[i, 0] = np.mean(preds)

    return rewards


def local_decompose_metrics(dataset_name, seed, interaction_root, script_dir):
    """Local implementation of decompose_metrics."""
    from tools.decomposition import (
        decompose_spd, decompose_eod, decompose_ped, decompose_aod,
        decompose_rvif, decompose_cacc, decompose_crec, decompose_cfpr
    )

    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    data_save_root = osp.join(script_dir, "../data/tabular", dataset_name,
                             "prepared_data", f"seed_{seed}")
    constraints = np.load(osp.join(data_save_root, "constraints.npy"))

    interactions = np.load(osp.join(interaction_root, "interactions.npy"))
    interactions = np.squeeze(interactions, axis=-1)
    rewards = np.load(osp.join(interaction_root, "rewards.npy"))
    rewards = np.squeeze(rewards, axis=-1)
    rewards_wo_mask = rewards[:, -1]
    data_analyzed = np.load(osp.join(interaction_root, "data_analyzed.npy"))
    labels = data_analyzed[:, -1]

    cv_root = osp.join(interaction_root, "contribution_vectors")
    makedirs(cv_root)

    metrics_computed = {}

    cacc, cacc_utic = decompose_cacc(labels, interactions, rewards_wo_mask)
    np.save(osp.join(cv_root, "accuracy_interaction_contribution.npy"), cacc_utic)
    metrics_computed["accuracy"] = 1 - cacc

    crec, crec_utic = decompose_crec(labels, interactions, rewards_wo_mask)
    np.save(osp.join(cv_root, "recall_interaction_contribution.npy"), crec_utic)
    metrics_computed["recall"] = 1 - crec

    cfpr, cfpr_utic = decompose_cfpr(labels, interactions, rewards_wo_mask)
    np.save(osp.join(cv_root, "FPR_interaction_contribution.npy"), cfpr_utic)
    metrics_computed["FPR"] = cfpr

    espd, espd_igid = decompose_spd(rewards_wo_mask, interactions,
                                    data_analyzed[:, :-1], sensitive_indices, constraints)
    np.save(osp.join(cv_root, "SPD_interaction_contribution.npy"), espd_igid)
    metrics_computed["SPD"] = espd

    eeod, eeod_igid = decompose_eod(labels, rewards_wo_mask, interactions,
                                    data_analyzed[:, :-1], sensitive_indices, constraints)
    np.save(osp.join(cv_root, "EOD_interaction_contribution.npy"), eeod_igid)
    metrics_computed["EOD"] = eeod

    eped, eped_igid = decompose_ped(labels, rewards_wo_mask, interactions,
                                    data_analyzed[:, :-1], sensitive_indices, constraints)
    np.save(osp.join(cv_root, "PED_interaction_contribution.npy"), eped_igid)
    metrics_computed["PED"] = eped

    eaod, eaod_igid = decompose_aod(labels, rewards_wo_mask, interactions,
                                    data_analyzed[:, :-1], sensitive_indices, constraints)
    np.save(osp.join(cv_root, "AOD_interaction_contribution.npy"), eaod_igid)
    metrics_computed["AOD"] = eaod

    for css, metric_name in [("CF", "CFVR"), ("GIF", "GIFVR")]:
        int1_path = osp.join(interaction_root, f"{css}_interactions_1.npy")
        int2_path = osp.join(interaction_root, f"{css}_interactions_2.npy")
        if not (osp.exists(int1_path) and osp.exists(int2_path)):
            continue
        int1 = np.squeeze(np.load(int1_path), axis=-1)
        int2 = np.squeeze(np.load(int2_path), axis=-1)
        rew1 = np.squeeze(np.load(osp.join(interaction_root, f"{css}_rewards_1.npy")), axis=-1)
        rew2 = np.squeeze(np.load(osp.join(interaction_root, f"{css}_rewards_2.npy")), axis=-1)
        rvif, rvif_utic = decompose_rvif(int1, int2, rew1[:, -1], rew2[:, -1])
        np.save(osp.join(cv_root, f"{metric_name}_interaction_contribution.npy"), rvif_utic)
        metrics_computed[metric_name] = rvif

    model_save_dir = osp.join(script_dir, "../models", dataset_name, "vanilla", "lr", f"seed_{seed}")
    with open(osp.join(model_save_dir, f"lr_estimated_metrics_marginal.json"), "w") as f:
        json.dump(metrics_computed, f, indent=1, cls=NumpyEncoder)

    return metrics_computed


def local_calculate_compatibility(interaction_root):
    """Compute pairwise cosine similarity."""
    cv_root = osp.join(interaction_root, "contribution_vectors")
    compat_root = osp.join(interaction_root, "compatibility")
    makedirs(compat_root)

    vectors = {}
    for fname in sorted(os.listdir(cv_root)):
        if fname.endswith("_interaction_contribution.npy"):
            metric_name = fname.replace("_interaction_contribution.npy", "")
            vectors[metric_name] = np.load(osp.join(cv_root, fname))

    metric_names = sorted(vectors.keys())
    compatibility = {}
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:
                v1, v2 = vectors[m1], vectors[m2]
                cos_sim, _ = cosine_similarity(v1, v2)
                compatibility[f"{m1}-{m2}"] = float(cos_sim)

    with open(osp.join(compat_root, "compatibility.json"), "w") as f:
        json.dump(compatibility, f, indent=2, cls=NumpyEncoder)

    return compatibility


def process_single_task(args):
    """Process a single (dataset, seed) combination."""
    dataset_name, seed, K, script_dir, project_root = args
    start_time = time.time()
    set_seed(seed)

    data_save_root = osp.join(script_dir, "../data/tabular", dataset_name,
                             "prepared_data", f"seed_{seed}")
    model_save_dir = osp.join(script_dir, "../models", dataset_name, "vanilla",
                             "lr", f"seed_{seed}")
    model_save_root = osp.join(model_save_dir, "lr")
    interaction_root = osp.join(model_save_dir, "interactions_marginal")
    mean_interaction_root = osp.join(model_save_dir, "interactions")

    compatibility_path = osp.join(interaction_root, "compatibility", "compatibility.json")
    if osp.exists(compatibility_path):
        return f"[SKIP] {dataset_name}/seed_{seed}"

    scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    data_analyzed = np.load(osp.join(data_save_root, "data_test_sampled.npy"))
    train_data = np.load(osp.join(data_save_root, "data_train.npy"))
    train_X = train_data[:, :-1]

    orig_model = load_classifier("lr", model_save_root, method_name="vanilla")
    model_callable, _ = model_standardize("lr", orig_model, scaler, "vanilla", sensitive_indices)

    n_players = data_analyzed.shape[1] - 1
    n_samples = data_analyzed.shape[0]

    makedirs(interaction_root)
    masks = generate_masks(n_players)
    rng = np.random.RandomState(seed)

    list_interactions = []
    list_rewards = []

    for i in range(n_samples):
        x = data_analyzed[i, :-1]
        rewards_i = compute_marginal_rewards(model_callable, x, train_X, masks, K=K, rng=rng)
        interactions_i = mobius_inversion(rewards_i)
        list_interactions.append(interactions_i)
        list_rewards.append(rewards_i)

    all_interactions = np.array(list_interactions)
    all_rewards = np.array(list_rewards)

    np.save(osp.join(interaction_root, "masks.npy"), masks)
    np.save(osp.join(interaction_root, "interactions.npy"), all_interactions)
    np.save(osp.join(interaction_root, "rewards.npy"), all_rewards)
    np.save(osp.join(interaction_root, "data_analyzed.npy"), data_analyzed)

    for css in ["CF", "GIF"]:
        pair1_path = osp.join(mean_interaction_root, f"{css}_pair_1.npy")
        pair2_path = osp.join(mean_interaction_root, f"{css}_pair_2.npy")
        if not (osp.exists(pair1_path) and osp.exists(pair2_path)):
            continue

        pair1_data = np.load(pair1_path)
        pair2_data = np.load(pair2_path)

        list_int_1, list_rew_1 = [], []
        list_int_2, list_rew_2 = [], []

        for i in range(n_samples):
            x1 = pair1_data[i, :-1]
            x2 = pair2_data[i, :-1]
            r1 = compute_marginal_rewards(model_callable, x1, train_X, masks, K=K, rng=rng)
            list_int_1.append(mobius_inversion(r1))
            list_rew_1.append(r1)
            r2 = compute_marginal_rewards(model_callable, x2, train_X, masks, K=K, rng=rng)
            list_int_2.append(mobius_inversion(r2))
            list_rew_2.append(r2)

        np.save(osp.join(interaction_root, f"{css}_pair_1.npy"), pair1_data)
        np.save(osp.join(interaction_root, f"{css}_pair_2.npy"), pair2_data)
        np.save(osp.join(interaction_root, f"{css}_interactions_1.npy"), np.array(list_int_1))
        np.save(osp.join(interaction_root, f"{css}_rewards_1.npy"), np.array(list_rew_1))
        np.save(osp.join(interaction_root, f"{css}_interactions_2.npy"), np.array(list_int_2))
        np.save(osp.join(interaction_root, f"{css}_rewards_2.npy"), np.array(list_rew_2))

    local_decompose_metrics(dataset_name, seed, interaction_root, script_dir)
    local_calculate_compatibility(interaction_root)

    elapsed = time.time() - start_time
    return f"[DONE] {dataset_name}/seed_{seed} in {elapsed:.1f}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=100, help='Number of strata (default: 100)')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        print("Running verification...")
        masks_ref = np.load(osp.join(project_root, "models/compas/vanilla/lr/seed_0/interactions/masks.npy"))
        data = np.load(osp.join(project_root, "models/compas/vanilla/lr/seed_0/interactions/data_analyzed.npy"))
        n_players = data.shape[1] - 1
        masks_test = generate_masks(n_players)
        print("[PASS] Masks encoding verified" if np.array_equal(masks_test, masks_ref) else "[FAIL]")
        sys.exit(0)

    datasets = DATASETS
    seeds = list(range(10))

    print("=" * 60)
    print("Marginal Sampling Experiment (K=100, Stratified Sampling)")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"K: {args.K}")
    print(f"CPU cores: {cpu_count()}")
    print("=" * 60)

    tasks = []
    for dataset in datasets:
        for seed in seeds:
            tasks.append((dataset, seed, args.K, script_dir, project_root))

    print(f"Total tasks: {len(tasks)}")

    n_workers = cpu_count()
    print(f"Using {n_workers} parallel workers")

    start_time = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(process_single_task, tasks)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Results:")
    for r in results:
        print(f"  {r}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
