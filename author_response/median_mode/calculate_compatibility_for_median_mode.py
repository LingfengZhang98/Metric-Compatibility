"""
Calculate Harsanyi interactions, decompose metrics, and calculate pairwise compatibility
for Median and Mode baselines using multiprocessing.
Extended to 6 datasets: census, ufrgs, compas, diabetes, bank, heart
"""

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
script_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.join(script_dir, "..")
sys.path.append(project_root)

import json
from itertools import combinations
import multiprocessing as mp
import numpy as np
import torch
torch.set_num_threads(4)
import joblib

from tools.config import RANDOM_SEED_LIST, considered_sensitive_attributes, list_group_fairness, list_individual_fairness, list_utilities
from tools.utils import makedirs, cosine_similarity, NumpyEncoder, set_seed, max_output_diff_pair, enumerate_subgroups
from tools.models import load_classifier, model_standardize, LogisticRegressionModel
from tools.evaluation import generate_similar_samples, generate_global_similar_samples_advanced_sampling
from tools.harsanyi.and_harsanyi import AndHarsanyi
from tools.harsanyi.harsanyi_utils import mask_input_fn_tabular

METRICS = ["accuracy", "recall", "FPR", "SPD", "EOD", "PED", "AOD", "CFVR", "GIFVR"]

DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
METHOD = "vanilla"
CLASSIFIER = "lr"
SEEDS = list(range(10))
BASELINE_TYPES = ["median", "mode"]


def calculate_harsanyi_interactions_with_baseline(
    dataset_name, method_name, classifier_name, seed, 
    baseline, baseline_name,
    cal_similar_samples=None, checkpoint_idx=None, hifi_eta=None
):
    print(f"Calculating Harsanyi interactions on {dataset_name} dataset with {method_name} method, {classifier_name} classifier, seed={seed}, baseline={baseline_name}\n.")
    set_seed(seed)
    
    data_save_root = osp.join(project_root, "data", "tabular", dataset_name, "prepared_data", f"seed_{seed}")
    scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    constraints = np.load(osp.join(data_save_root, "constraints.npy"))
    data_analyzed = np.load(osp.join(data_save_root, "data_test_sampled.npy"))
    
    model_save_dir = osp.join(project_root, "models", dataset_name, method_name, classifier_name, f"seed_{seed}")
    model_save_root = osp.join(model_save_dir, classifier_name)
    
    if checkpoint_idx is None:
        if hifi_eta is None:
            orig_model = load_classifier(classifier_name, model_save_root, method_name=method_name)
        elif method_name == "hifi" and classifier_name == "lr":
            orig_model = LogisticRegressionModel(data_analyzed.shape[1]-1)
            orig_model.load_state_dict(torch.load(model_save_root+f"_eta={hifi_eta}.pth"))
    else:
        orig_model = load_classifier(classifier_name, model_save_root, checkpoint_idx, method_name)
    
    if hifi_eta is None:
        model, reward_type = model_standardize(classifier_name, orig_model, scaler, method_name, sensitive_indices)
    elif method_name == "hifi" and classifier_name == "lr":
        from functools import partial
        def get_hifi_output(X, orig_model):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            orig_model.to(device)
            with torch.no_grad():
                X_scaled = scaler.transform(X)
                X_scaled = torch.from_numpy(X_scaled).float().to(device)
                return orig_model(X_scaled).cpu().numpy().reshape(-1, 1)
        model = partial(get_hifi_output, orig_model=orig_model)
        reward_type = "positive_probability"
    
    if checkpoint_idx is None:
        if hifi_eta is None:
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}")
        elif method_name == "hifi" and classifier_name == "lr":
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_eta={hifi_eta}")
    else:
        interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_checkpoint_{checkpoint_idx}")
    makedirs(interaction_root)

    if cal_similar_samples is None:
        cal_similar_samples = []
    elif isinstance(cal_similar_samples, str):
        cal_similar_samples = [cal_similar_samples]
    elif isinstance(cal_similar_samples, (list, tuple)):
        cal_similar_samples = list(cal_similar_samples)

    if method_name == "blindness":
        data_analyzed_with_SA = data_analyzed.copy()
        data_analyzed = np.delete(data_analyzed, sensitive_indices, axis=1)
        baseline = np.delete(baseline, sensitive_indices, axis=1)
        constraints = np.delete(constraints, sensitive_indices, axis=0)
        sensitive_indices = []
        np.save(osp.join(interaction_root, "data_analyzed_with_SA.npy"), data_analyzed_with_SA)
    np.save(osp.join(interaction_root, "data_analyzed.npy"), data_analyzed)
    print(f"Total number of samples: {data_analyzed.shape[0]}")
    print(f"Total number of features: {data_analyzed.shape[1]-1}\n")

    masks = np.array([])
    list_interactions = []
    list_rewards = []
    for i in range(data_analyzed.shape[0]):
        x = data_analyzed[i, :-1]
        y = data_analyzed[i, -1]
        calculator = AndHarsanyi(
            model=model, reward_type=reward_type, x=x.reshape(1, -1), y=y,
            baseline=baseline, all_players=list(range(len(x))),
            mask_input_fn=mask_input_fn_tabular
        )
        calculator.attribute()
        if i == 0:
            masks = calculator.get_masks()
        interactions = calculator.get_interaction()
        rewards = calculator.get_rewards()
        list_interactions.append(interactions)
        list_rewards.append(rewards)

    np.save(osp.join(interaction_root, "masks.npy"), masks)
    np.save(osp.join(interaction_root, "interactions.npy"), np.array(list_interactions))
    np.save(osp.join(interaction_root, "rewards.npy"), np.array(list_rewards))

    for css in cal_similar_samples:
        if css not in ["CF", "GIF"]:
            continue
        list_max_diff_pair_1, list_interactions_1, list_rewards_1 = [], [], []
        list_max_diff_pair_2, list_interactions_2, list_rewards_2 = [], [], []

        for i in range(data_analyzed.shape[0]):
            print(f"Processing [{css}] sample [{i}] on {dataset_name}-{method_name}-{classifier_name}-{seed}-{baseline_name}\n")
            x = data_analyzed[i, :-1]
            y = data_analyzed[i, -1]
            if css == "CF":
                similar_x = generate_similar_samples(x, sensitive_indices, constraints)
            elif css == "GIF":
                similar_x = generate_global_similar_samples_advanced_sampling(x, constraints, seed=seed, sensitive_indices=sensitive_indices)
            pair1, pair2 = max_output_diff_pair(similar_x, model)
            list_max_diff_pair_1.append(pair1)
            list_max_diff_pair_2.append(pair2)

            for pair_x in [pair1, pair2]:
                calculator = AndHarsanyi(
                    model=model, reward_type=reward_type, x=pair_x.reshape(1, -1), y=y,
                    baseline=baseline, all_players=list(range(len(x))),
                    mask_input_fn=mask_input_fn_tabular
                )
                calculator.attribute()
                interactions = calculator.get_interaction()
                rewards = calculator.get_rewards()
                if pair_x is pair1:
                    list_interactions_1.append(interactions)
                    list_rewards_1.append(rewards)
                else:
                    list_interactions_2.append(interactions)
                    list_rewards_2.append(rewards)

        np.save(osp.join(interaction_root, css + "_pair_1.npy"), np.column_stack((np.array(list_max_diff_pair_1), data_analyzed[:, -1])))
        np.save(osp.join(interaction_root, css + "_interactions_1.npy"), np.array(list_interactions_1))
        np.save(osp.join(interaction_root, css + "_rewards_1.npy"), np.array(list_rewards_1))
        np.save(osp.join(interaction_root, css + "_pair_2.npy"), np.column_stack((np.array(list_max_diff_pair_2), data_analyzed[:, -1])))
        np.save(osp.join(interaction_root, css + "_interactions_2.npy"), np.array(list_interactions_2))
        np.save(osp.join(interaction_root, css + "_rewards_2.npy"), np.array(list_rewards_2))


def decompose_spd(pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)
    probabilities, masks = [], []
    for combination in all_combinations:
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        if np.sum(mask) > 0:
            probabilities.append(np.mean(pre_prob[mask]))
            masks.append(mask)
    probabilities = np.array(probabilities)
    espd = np.max(probabilities) - np.min(probabilities)
    espd_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)
    return espd, espd_igid

def decompose_aod(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)
    sums, masks = [], []
    for combination in all_combinations:
        base_mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            base_mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask_y0 = base_mask & (y_true.flatten() == 0)
        mask_y1 = base_mask & (y_true.flatten() == 1)
        prob_y0 = np.mean(pre_prob[mask_y0]) if np.sum(mask_y0) > 0 else 0.0
        prob_y1 = np.mean(pre_prob[mask_y1]) if np.sum(mask_y1) > 0 else 0.0
        sums.append(prob_y0 + prob_y1)
        masks.append((mask_y0, mask_y1))
    sums = np.array(sums)
    eaod = 0.5 * (np.max(sums) - np.min(sums))
    idx_max, idx_min = np.argmax(sums), np.argmin(sums)
    igid_y0_p = np.mean(interactions[masks[idx_max][0]], axis=0) if np.sum(masks[idx_max][0]) > 0 else np.zeros(interactions.shape[1])
    igid_y1_p = np.mean(interactions[masks[idx_max][1]], axis=0) if np.sum(masks[idx_max][1]) > 0 else np.zeros(interactions.shape[1])
    igid_y0_up = np.mean(interactions[masks[idx_min][0]], axis=0) if np.sum(masks[idx_min][0]) > 0 else np.zeros(interactions.shape[1])
    igid_y1_up = np.mean(interactions[masks[idx_min][1]], axis=0) if np.sum(masks[idx_min][1]) > 0 else np.zeros(interactions.shape[1])
    eaod_igid = 0.5 * (igid_y0_p + igid_y1_p - igid_y0_up - igid_y1_up)
    return eaod, eaod_igid

def decompose_eod(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)
    probabilities, masks = [], []
    for combination in all_combinations:
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 1)
        if np.sum(mask) > 0:
            probabilities.append(np.mean(pre_prob[mask]))
            masks.append(mask)
    probabilities = np.array(probabilities)
    eeod = np.max(probabilities) - np.min(probabilities)
    eeod_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)
    return eeod, eeod_igid

def decompose_ped(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)
    probabilities, masks = [], []
    for combination in all_combinations:
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 0)
        if np.sum(mask) > 0:
            probabilities.append(np.mean(pre_prob[mask]))
            masks.append(mask)
    probabilities = np.array(probabilities)
    eped = np.max(probabilities) - np.min(probabilities)
    eped_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)
    return eped, eped_igid

def decompose_rvif(interactions1, interactions2, pre_prob1, pre_prob2):
    rvif = np.mean(pre_prob2 - pre_prob1)
    rvif_utic = np.mean(interactions2 - interactions1, axis=0)
    return rvif, rvif_utic

def decompose_cacc(labels, interactions, pre_prob):
    favorable_label_ratio = np.mean(labels==1)
    cacc = 1 - (np.sum(pre_prob[labels==1]) + np.sum(1 - pre_prob[labels==0])) / len(labels)
    cacc_utic = - (np.sum(interactions[labels==1], axis=0) - np.sum(interactions[labels==0], axis=0)) / len(labels)
    cacc_utic[0] += favorable_label_ratio
    return cacc, cacc_utic

def decompose_crec(labels, interactions, pre_prob):
    crec = 1 - np.mean(pre_prob[labels==1])
    crec_utic = - np.mean(interactions[labels==1], axis=0)
    crec_utic[0] += 1
    return crec, crec_utic

def decompose_cfpr(labels, interactions, pre_prob):
    cfpr = np.mean(pre_prob[labels==0])
    cfpr_utic = np.mean(interactions[labels==0], axis=0)
    return cfpr, cfpr_utic


def decompose_metrics_with_baseline(dataset_name, method_name, classifier_name, seed, baseline_name, checkpoint_idx=None, hifi_eta=None, metrics=METRICS):
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    model_save_dir = osp.join(project_root, "models", dataset_name, method_name, classifier_name, f"seed_{seed}")
    
    if checkpoint_idx is None:
        if hifi_eta is None:
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}")
        elif method_name == "hifi" and classifier_name == "lr":
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_eta={hifi_eta}")
    else:
        interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_checkpoint_{checkpoint_idx}")
    
    contribution_root = osp.join(interaction_root, "contribution_vectors")
    makedirs(contribution_root)
    data_save_root = osp.join(project_root, "data", "tabular", dataset_name, "prepared_data", f"seed_{seed}")

    if method_name == "blindness":
        data_analyzed = np.load(osp.join(interaction_root, "data_analyzed_with_SA.npy"))
    else:
        data_analyzed = np.load(osp.join(interaction_root, "data_analyzed.npy"))
    
    interactions = np.load(osp.join(interaction_root, "interactions.npy"))
    interactions = np.squeeze(interactions, axis=-1)
    rewards = np.load(osp.join(interaction_root, "rewards.npy"))
    rewards = np.squeeze(rewards, axis=-1)
    rewards_wo_mask = rewards[:, -1]
    constraints = np.load(osp.join(data_save_root, "constraints.npy"))

    local_vars = {"cacc": None, "crec": None, "cfpr": None, "espd": None, "eeod": None, "eped": None, "eaod": None, "rvcf": None, "rvgif": None}
    for metric in metrics:
        if metric not in list_group_fairness + list_individual_fairness + list_utilities:
            continue

        if metric == "accuracy":
            cacc, cacc_utic = decompose_cacc(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), cacc_utic)
            local_vars["cacc"] = cacc
        elif metric == "recall":
            crec, crec_utic = decompose_crec(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), crec_utic)
            local_vars["crec"] = crec
        elif metric == "FPR":
            cfpr, cfpr_utic = decompose_cfpr(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), cfpr_utic)
            local_vars["cfpr"] = cfpr
        elif metric == "SPD":
            espd, espd_igid = decompose_spd(rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), espd_igid)
            local_vars["espd"] = espd
        elif metric == "EOD":
            eeod, eeod_igid = decompose_eod(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eeod_igid)
            local_vars["eeod"] = eeod
        elif metric == "PED":
            eped, eped_igid = decompose_ped(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eped_igid)
            local_vars["eped"] = eped
        elif metric == "AOD":
            eaod, eaod_igid = decompose_aod(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eaod_igid)
            local_vars["eaod"] = eaod
        elif metric == "CFVR":
            CF_interactions_1 = np.squeeze(np.load(osp.join(interaction_root, "CF_interactions_1.npy")), axis=-1)
            CF_interactions_2 = np.squeeze(np.load(osp.join(interaction_root, "CF_interactions_2.npy")), axis=-1)
            CF_rewards_1 = np.squeeze(np.load(osp.join(interaction_root, "CF_rewards_1.npy")), axis=-1)
            CF_rewards_2 = np.squeeze(np.load(osp.join(interaction_root, "CF_rewards_2.npy")), axis=-1)
            rvcf, rvcf_utic = decompose_rvif(CF_interactions_1, CF_interactions_2, CF_rewards_1[:, -1], CF_rewards_2[:, -1])
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), rvcf_utic)
            local_vars["rvcf"] = rvcf
        elif metric == "GIFVR":
            GIF_interactions_1 = np.squeeze(np.load(osp.join(interaction_root, "GIF_interactions_1.npy")), axis=-1)
            GIF_interactions_2 = np.squeeze(np.load(osp.join(interaction_root, "GIF_interactions_2.npy")), axis=-1)
            GIF_rewards_1 = np.squeeze(np.load(osp.join(interaction_root, "GIF_rewards_1.npy")), axis=-1)
            GIF_rewards_2 = np.squeeze(np.load(osp.join(interaction_root, "GIF_rewards_2.npy")), axis=-1)
            rvgif, rvgif_utic = decompose_rvif(GIF_interactions_1, GIF_interactions_2, GIF_rewards_1[:, -1], GIF_rewards_2[:, -1])
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), rvgif_utic)
            local_vars["rvgif"] = rvgif

    if hifi_eta is None:
        model_metrics = {
            "utilities": {"accuracy": 1 - local_vars["cacc"], "recall": 1 - local_vars["crec"], "FPR": local_vars["cfpr"]},
            "fairness": {"SPD": local_vars["espd"], "EOD": local_vars["eeod"], "PED": local_vars["eped"], "AOD": local_vars["eaod"], "CFVR": local_vars["rvcf"], "GIFVR": local_vars["rvgif"]}
        }
        with open(osp.join(model_save_dir, f"{classifier_name}_estimated_metrics_{baseline_name}.json"), "w") as f:
            json.dump(model_metrics, f, indent=1, cls=NumpyEncoder)


def calculate_compatibility_with_baseline(dataset_name, method_name, classifier_name, seed, baseline_name, checkpoint_idx=None, hifi_eta=None, metrics=METRICS):
    model_save_dir = osp.join(project_root, "models", dataset_name, method_name, classifier_name, f"seed_{seed}")
    if checkpoint_idx is None:
        if hifi_eta is None:
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}")
        elif method_name == "hifi" and classifier_name == "lr":
            interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_eta={hifi_eta}")
    else:
        interaction_root = osp.join(model_save_dir, f"interactions_{baseline_name}_checkpoint_{checkpoint_idx}")
    
    contribution_root = osp.join(interaction_root, "contribution_vectors")
    compatibility_dir = osp.join(interaction_root, "compatibility")
    makedirs(compatibility_dir)
    
    contribution_vectors = {metric: np.load(osp.join(contribution_root, f"{metric}_interaction_contribution.npy")) for metric in metrics}
    
    metric_pairs = list(combinations(metrics, 2))
    compatibility = {}
    for pair in metric_pairs:
        compatibility[f"{pair[0]}-{pair[1]}"], component_vector = cosine_similarity(contribution_vectors[pair[0]], contribution_vectors[pair[1]])
        np.save(osp.join(compatibility_dir, f"{pair[0]}-{pair[1]}_components.npy"), component_vector)
    
    with open(osp.join(compatibility_dir, "compatibility.json"), "w") as f:
        json.dump(compatibility, f, indent=1, cls=NumpyEncoder)


def worker_task(args):
    dataset_name, method_name, classifier_name, seed, baseline_path, baseline_name = args
    
    baseline = np.load(baseline_path)
    if baseline.ndim == 1:
        baseline = baseline.reshape(1, -1)
    
    calculate_harsanyi_interactions_with_baseline(
        dataset_name, method_name, classifier_name, seed,
        baseline=baseline, baseline_name=baseline_name,
        cal_similar_samples=["CF", "GIF"]
    )
    
    decompose_metrics_with_baseline(
        dataset_name, method_name, classifier_name, seed,
        baseline_name=baseline_name
    )
    
    calculate_compatibility_with_baseline(
        dataset_name, method_name, classifier_name, seed,
        baseline_name=baseline_name
    )
    
    return f"Done: {dataset_name}-{method_name}-{classifier_name}-{seed}-{baseline_name}"


if __name__ == '__main__':
    tasks = []
    for DATASET in DATASETS:
        for seed in SEEDS:
            for baseline_name in BASELINE_TYPES:
                baseline_path = osp.join(script_dir, DATASET, f"seed_{seed}", f"{baseline_name}_baseline.npy")
                tasks.append((DATASET, METHOD, CLASSIFIER, seed, baseline_path, baseline_name))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Datasets: {DATASETS}")
    print(f"Method: {METHOD}, Classifier: {CLASSIFIER}")
    print(f"Baselines: {BASELINE_TYPES}, Seeds: {SEEDS}")
    print(f"Metrics: {METRICS}")
    print(f"Using {mp.cpu_count()} CPU cores")
    print("=" * 80)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(worker_task, tasks)
    
    print("=" * 80)
    print("All tasks completed!")
    for r in results:
        print(r)
