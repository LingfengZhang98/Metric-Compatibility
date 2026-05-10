import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import json
import numpy as np
import joblib
import torch

from .config import considered_sensitive_attributes, list_utilities, list_group_fairness, list_individual_fairness, METRICS
from .utils import set_seed, makedirs, max_output_diff_pair, enumerate_subgroups, NumpyEncoder
from .evaluation import generate_similar_samples, generate_global_similar_samples_advanced_sampling
from .models import load_classifier, model_standardize, LogisticRegressionModel
from .harsanyi.and_harsanyi import AndHarsanyi
from .harsanyi.harsanyi_utils import mask_input_fn_tabular


def calculate_harsanyi_interactions(dataset_name, method_name, classifier_name, seed, cal_similar_samples=["CF", "GIF"], checkpoint_idx=None, hifi_eta=None):
    print(f"Calculating harsanyi interactions on {dataset_name} dataset with {method_name} method, {classifier_name} classifier, and seed={seed}\n.")
    set_seed(seed)
    
    # load data
    data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")
    scaler = joblib.load(osp.join(data_save_root, "scaler.pkl"))
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    baseline = np.load(osp.join(data_save_root, "baseline.npy"))
    constraints = np.load(osp.join(data_save_root, "constraints.npy"))
    data_analyzed = np.load(osp.join(data_save_root, "data_test_sampled.npy"))
    
    # load model
    model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
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
            interaction_root = osp.join(model_save_dir, "interactions")
        elif method_name == "hifi" and classifier_name == "lr":
            interaction_root = osp.join(model_save_dir, f"interactions_eta={hifi_eta}")
    else:
        interaction_root = osp.join(model_save_dir, f"interactions_checkpoint_{checkpoint_idx}")
    makedirs(interaction_root)

    if cal_similar_samples is None:
        cal_similar_samples = []
    elif isinstance(cal_similar_samples, str):
        cal_similar_samples = [cal_similar_samples]
    elif isinstance(cal_similar_samples, (list, tuple)):
        cal_similar_samples = list(cal_similar_samples)
    else:
        raise TypeError(f"cal_similar_samples must be a string, list, tuple, or None, but got: {type(cal_similar_samples)}")

    # prepare data for analysis
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

    # calculate interactions on data to be analyzed
    masks = np.array([])
    list_interactions = []
    list_rewards = []
    for i in range(data_analyzed.shape[0]):
        print(f"Processing the sample [{i}] on {dataset_name}-{method_name}-{classifier_name}-{seed}\n")
        x = data_analyzed[i, :-1]
        y = data_analyzed[i, -1]
        calculator = AndHarsanyi(
            model=model,
            reward_type=reward_type,
            x=x.reshape(1, -1),
            y=y,
            baseline=baseline,
            all_players=list(range(len(x))),
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

    # calculate interactions on similar samples w.r.t. Causal Fairness or Global Individual Fairness
    for css in cal_similar_samples:
        if css not in ["CF", "GIF"]:
            continue

        list_max_diff_pair_1 = []  # instances whose predicted probability is lower
        list_interactions_1 = []
        list_rewards_1 = []
        list_max_diff_pair_2 = []  # instances whose predicted probability is higher
        list_interactions_2 = []
        list_rewards_2 = []

        for i in range(data_analyzed.shape[0]):
            print(f"Processing the sample [{i}] for [{css}] on {dataset_name}-{method_name}-{classifier_name}-{seed}\n")
            x = data_analyzed[i, :-1]
            y = data_analyzed[i, -1]
            if css == "CF":
                similar_x = generate_similar_samples(x, sensitive_indices, constraints)
            elif css == "GIF":
                similar_x = generate_global_similar_samples_advanced_sampling(x, constraints, seed=seed, sensitive_indices=sensitive_indices)
            pair1, pair2 = max_output_diff_pair(similar_x, model)
            list_max_diff_pair_1.append(pair1)
            list_max_diff_pair_2.append(pair2)

            calculator = AndHarsanyi(
                model=model,
                reward_type=reward_type,
                x=pair1.reshape(1, -1),
                y=y,
                baseline=baseline,
                all_players=list(range(len(x))),
                mask_input_fn=mask_input_fn_tabular
            )
            calculator.attribute()
            interactions = calculator.get_interaction()
            rewards = calculator.get_rewards()
            list_interactions_1.append(interactions)
            list_rewards_1.append(rewards)

            calculator = AndHarsanyi(
                model=model,
                reward_type=reward_type,
                x=pair2.reshape(1, -1),
                y=y,
                baseline=baseline,
                all_players=list(range(len(x))),
                mask_input_fn=mask_input_fn_tabular
            )
            calculator.attribute()
            interactions = calculator.get_interaction()
            rewards = calculator.get_rewards()
            list_interactions_2.append(interactions)
            list_rewards_2.append(rewards)

        np.save(osp.join(interaction_root, css + "_pair_1.npy"), np.column_stack((np.array(list_max_diff_pair_1), data_analyzed[:, -1])))
        np.save(osp.join(interaction_root, css + "_interactions_1.npy"), np.array(list_interactions_1))
        np.save(osp.join(interaction_root, css + "_rewards_1.npy"), np.array(list_rewards_1))
        np.save(osp.join(interaction_root, css + "_pair_2.npy"), np.column_stack((np.array(list_max_diff_pair_2), data_analyzed[:, -1])))
        np.save(osp.join(interaction_root, css + "_interactions_2.npy"), np.array(list_interactions_2))
        np.save(osp.join(interaction_root, css + "_rewards_2.npy"), np.array(list_rewards_2))


def decompose_spd(pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    """
    Decompose the extended statistical parity difference (ESPD) into the sum of inter-group interaction differences (IGIDs).
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    masks = []
    for combination in all_combinations:
        # Create mask for current subgroup
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])

        # Calculate E[v(x)|A=s]
        if np.sum(mask) > 0:
            prob = np.mean(pre_prob[mask])
            probabilities.append(prob)
            masks.append(mask)

    probabilities = np.array(probabilities)
    espd = np.max(probabilities) - np.min(probabilities)
    espd_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)

    return espd, espd_igid


def decompose_aod(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    """
    Decompose the extended average odds difference (EAOD) into the sum of inter-group interaction differences (IGIDs).
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    sums = []
    masks = []
    for combination in all_combinations:
        # Create base mask for current subgroup
        base_mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            base_mask &= (sensitive_data[:, attr_idx] == combination[i])

        # Calculate E[v(x)|A=s,Y=0]
        mask_y0 = base_mask & (y_true.flatten() == 0)
        if np.sum(mask_y0) > 0:
            prob_y0 = np.mean(pre_prob[mask_y0])
        else:
            prob_y0 = 0.0

        # Calculate E[v(x)|A=s,Y=1]
        mask_y1 = base_mask & (y_true.flatten() == 1)
        if np.sum(mask_y1) > 0:
            prob_y1 = np.mean(pre_prob[mask_y1])
        else:
            prob_y1 = 0.0

        sums.append(prob_y0 + prob_y1)
        masks.append((mask_y0, mask_y1))

    sums = np.array(sums)
    eaod = 0.5 * (np.max(sums) - np.min(sums))

    tuple_privileged_group = masks[int(np.argmax(sums))]
    if np.sum(tuple_privileged_group[0]) > 0:
        interaction_y0_p = np.mean(interactions[tuple_privileged_group[0]], axis=0)
    else:
        interaction_y0_p = np.zeros(interactions.shape[1])
    if np.sum(tuple_privileged_group[1]) > 0:
        interaction_y1_p = np.mean(interactions[tuple_privileged_group[1]], axis=0)
    else:
        interaction_y1_p = np.zeros(interactions.shape[1])

    tuple_unprivileged_group = masks[int(np.argmin(sums))]
    if np.sum(tuple_unprivileged_group[0]) > 0:
        interaction_y0_up = np.mean(interactions[tuple_unprivileged_group[0]], axis=0)
    else:
        interaction_y0_up = np.zeros(interactions.shape[1])
    if np.sum(tuple_unprivileged_group[1]) > 0:
        interaction_y1_up = np.mean(interactions[tuple_unprivileged_group[1]], axis=0)
    else:
        interaction_y1_up = np.zeros(interactions.shape[1])

    eaod_igid = 0.5 * (interaction_y0_p + interaction_y1_p - interaction_y0_up - interaction_y1_up)

    return eaod, eaod_igid


def decompose_eod(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    """
    Decompose the extended equal opportunity difference (EEOD) into the sum of inter-group interaction differences (IGIDs).
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    masks = []
    for combination in all_combinations:
        # Create mask for current subgroup with Y=1
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 1)

        # Calculate E[v(x)|A=s,Y=1]
        if np.sum(mask) > 0:
            prob = np.mean(pre_prob[mask])
            probabilities.append(prob)
            masks.append(mask)

    probabilities = np.array(probabilities)
    eeod = np.max(probabilities) - np.min(probabilities)
    eeod_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)

    return eeod, eeod_igid


def decompose_ped(y_true, pre_prob, interactions, sensitive_data, sensitive_indices, value_ranges):
    """
    Decompose the extended predictive equality difference (EPED) into the sum of inter-group interaction differences (IGIDs).
    """
    # Generate all combinations
    all_combinations = enumerate_subgroups(sensitive_indices, value_ranges)

    probabilities = []
    masks = []
    for combination in all_combinations:
        # Create mask for current subgroup with Y=0
        mask = np.ones(len(sensitive_data), dtype=bool)
        for i, attr_idx in enumerate(sensitive_indices):
            mask &= (sensitive_data[:, attr_idx] == combination[i])
        mask &= (y_true.flatten() == 0)

        # Calculate E[v(x)|A=s,Y=0]
        if np.sum(mask) > 0:
            prob = np.mean(pre_prob[mask])
            probabilities.append(prob)
            masks.append(mask)

    probabilities = np.array(probabilities)
    eped = np.max(probabilities) - np.min(probabilities)
    eped_igid = np.mean(interactions[masks[int(np.argmax(probabilities))]], axis=0) - np.mean(interactions[masks[int(np.argmin(probabilities))]], axis=0)

    return eped, eped_igid


def decompose_rvif(interactions1, interactions2, pre_prob1, pre_prob2):
    """
    Compute the risk of violating individual fairness (RVIF), e.g, RVCF and RVGIF, and its inter-individual interaction differences (IIIDs).
    """
    rvif = np.mean(pre_prob2 - pre_prob1)
    rvif_utic = np.mean(interactions2 - interactions1, axis=0)

    return rvif, rvif_utic


def decompose_cacc(labels, interactions, pre_prob):
    """
    Compute the continuous accuracy (CAcc, reversed) and its utility-targeted interaction contributions (UTICs).
    """
    favorable_label_ratio = np.mean(labels==1)
    cacc = 1 - (np.sum(pre_prob[labels==1]) + np.sum(1 - pre_prob[labels==0])) / len(labels)
    cacc_utic = - (np.sum(interactions[labels==1], axis=0) - np.sum(interactions[labels==0], axis=0)) / len(labels)
    cacc_utic[0] += favorable_label_ratio

    return cacc, cacc_utic


def decompose_crec(labels, interactions, pre_prob):
    """
    Compute the continuous recall (CRec, reversed) and its utility-targeted interaction contributions (UTICs).
    """
    crec = 1 - np.mean(pre_prob[labels==1])
    crec_utic = - np.mean(interactions[labels==1], axis=0)
    crec_utic[0] += 1

    return crec, crec_utic


def decompose_cfpr(labels, interactions, pre_prob):
    """
    Compute the continuous false positive rate (CFPR) and its utility-targeted interaction contributions (UTICs).
    """
    cfpr = np.mean(pre_prob[labels==0])
    cfpr_utic = np.mean(interactions[labels==0], axis=0)

    return cfpr, cfpr_utic


def decompose_metrics(dataset_name, method_name, classifier_name, seed, checkpoint_idx=None, hifi_eta=None, metrics=METRICS):
    sensitive_indices = list(considered_sensitive_attributes[dataset_name].values())
    
    model_save_dir = osp.join(script_dir, "../models", dataset_name, method_name, 
                                   classifier_name, f"seed_{seed}")
    if checkpoint_idx is None:
        if hifi_eta is None:
            interaction_root = osp.join(model_save_dir, "interactions")
        elif method_name == "hifi" and classifier_name == "lr":
            interaction_root = osp.join(model_save_dir, f"interactions_eta={hifi_eta}")
    else:
        interaction_root = osp.join(model_save_dir, f"interactions_checkpoint_{checkpoint_idx}")
    contribution_root = osp.join(interaction_root, "contribution_vectors")
    makedirs(contribution_root)
    data_save_root = osp.join(script_dir, "../data/tabular", dataset_name, "prepared_data", f"seed_{seed}")

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

    for metric in metrics:
        if metric not in list_group_fairness and metric not in list_individual_fairness and metric not in list_utilities:
            raise NotImplementedError(f"Decomposition of [{metric}] has not been implemented.")

        elif metric == "accuracy":      # reversed by (1-x) for unified optimization direction
            cacc, cacc_utic = decompose_cacc(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), cacc_utic)

        elif metric == "recall":        # reversed by (1-x) for unified optimization direction
            crec, crec_utic = decompose_crec(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), crec_utic)

        elif metric == "FPR":
            cfpr, cfpr_utic = decompose_cfpr(data_analyzed[:, -1], interactions, rewards_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), cfpr_utic)

        elif metric == "SPD":
            espd, espd_igid = decompose_spd(rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), espd_igid)

        elif metric == "EOD":
            eeod, eeod_igid = decompose_eod(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eeod_igid)

        elif metric == "PED":
            eped, eped_igid = decompose_ped(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eped_igid)

        elif metric == "AOD":
            eaod, eaod_igid = decompose_aod(data_analyzed[:, -1], rewards_wo_mask, interactions, data_analyzed[:, :-1], sensitive_indices, constraints)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), eaod_igid)

        elif metric == "CFVR":
            CF_interactions_1 = np.load(osp.join(interaction_root, "CF_interactions_1.npy"))
            CF_interactions_1 = np.squeeze(CF_interactions_1, axis=-1)
            CF_interactions_2 = np.load(osp.join(interaction_root, "CF_interactions_2.npy"))
            CF_interactions_2 = np.squeeze(CF_interactions_2, axis=-1)
            CF_rewards_1 = np.load(osp.join(interaction_root, "CF_rewards_1.npy"))
            CF_rewards_1 = np.squeeze(CF_rewards_1, axis=-1)
            CF_rewards_1_wo_mask = CF_rewards_1[:, -1]
            CF_rewards_2 = np.load(osp.join(interaction_root, "CF_rewards_2.npy"))
            CF_rewards_2 = np.squeeze(CF_rewards_2, axis=-1)
            CF_rewards_2_wo_mask = CF_rewards_2[:, -1]
            rvcf, rvcf_utic = decompose_rvif(CF_interactions_1, CF_interactions_2, CF_rewards_1_wo_mask, CF_rewards_2_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), rvcf_utic)

        elif metric == "GIFVR":
            GIF_interactions_1 = np.load(osp.join(interaction_root, "GIF_interactions_1.npy"))
            GIF_interactions_1 = np.squeeze(GIF_interactions_1, axis=-1)
            GIF_interactions_2 = np.load(osp.join(interaction_root, "GIF_interactions_2.npy"))
            GIF_interactions_2 = np.squeeze(GIF_interactions_2, axis=-1)
            GIF_rewards_1 = np.load(osp.join(interaction_root, "GIF_rewards_1.npy"))
            GIF_rewards_1 = np.squeeze(GIF_rewards_1, axis=-1)
            GIF_rewards_1_wo_mask = GIF_rewards_1[:, -1]
            GIF_rewards_2 = np.load(osp.join(interaction_root, "GIF_rewards_2.npy"))
            GIF_rewards_2 = np.squeeze(GIF_rewards_2, axis=-1)
            GIF_rewards_2_wo_mask = GIF_rewards_2[:, -1]
            rvgif, rvgif_utic = decompose_rvif(GIF_interactions_1, GIF_interactions_2, GIF_rewards_1_wo_mask, GIF_rewards_2_wo_mask)
            np.save(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"), rvgif_utic)
    
    # Save estimated metrics
    if hifi_eta is None:
        model_metrics = {}
        model_metrics["utilities"] = {
            "accuracy": 1 - cacc,
            "recall": 1 - crec,
            "FPR": cfpr
        }
        model_metrics["fairness"] = {
            "SPD": espd,
            "EOD": eeod,
            "PED": eped,
            "AOD": eaod,
            "CFVR": rvcf,
            "GIFVR": rvgif
        }
        with open(osp.join(model_save_dir, f"{classifier_name}_estimated_metrics.json"), "w") as f:
            json.dump(model_metrics, f, indent=1, cls=NumpyEncoder)