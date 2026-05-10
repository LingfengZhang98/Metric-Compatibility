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

import json
from itertools import combinations
import multiprocessing as mp
import numpy as np
import torch
torch.set_num_threads(4)

from tools.decomposition import calculate_harsanyi_interactions, decompose_metrics
from tools.config import DATASETS, METHODS, MODELS, RANDOM_SEED_LIST, NUM_CHECKPOINTS, METRICS
from tools.utils import makedirs, cosine_similarity, NumpyEncoder


def calculate_compatibility(dataset_name, method_name, classifier_name, seed, checkpoint_idx=None, hifi_eta=None, metrics=METRICS):
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
    compatibility_dir = osp.join(interaction_root, "compatibility")
    makedirs(compatibility_dir)
    
    contribution_vectors = {}
    for metric in metrics:
        contribution_vectors[metric] = np.load(osp.join(contribution_root, f"{metric}_interaction_contribution.npy"))
    
    metric_pairs = list(combinations(metrics, 2))
    compatibility = {}
    for pair in metric_pairs:
        compatibility[f"{pair[0]}-{pair[1]}"], component_vector = cosine_similarity(contribution_vectors[pair[0]], contribution_vectors[pair[1]])
        np.save(osp.join(compatibility_dir, f"{pair[0]}-{pair[1]}_components.npy"), component_vector)
    
    with open(osp.join(compatibility_dir, "compatibility.json"), "w") as f:
        json.dump(compatibility, f, indent=1, cls=NumpyEncoder)


if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        for dataset in DATASETS:
            for method in METHODS:
                for classifier in MODELS:
                    for seed in RANDOM_SEED_LIST:
                        pool.apply_async(
                            calculate_harsanyi_interactions, 
                            args=(dataset, method, classifier, seed)
                        )
        pool.close()
        pool.join()
    
    with mp.Pool(processes=mp.cpu_count()//2) as pool:
        for seed in RANDOM_SEED_LIST:
            for idx in list(range(NUM_CHECKPOINTS)):
                pool.apply_async(
                    calculate_harsanyi_interactions,
                    args=("census", "vanilla", "tabnet", seed),
                    kwds={'checkpoint_idx': idx}
                )
        pool.close()
        pool.join()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for dataset in DATASETS:
            for method in METHODS:
                for classifier in MODELS:
                    for seed in RANDOM_SEED_LIST:
                        pool.apply_async(
                            decompose_metrics, 
                            args=(dataset, method, classifier, seed)
                        )
        pool.close()
        pool.join()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for seed in RANDOM_SEED_LIST:
            for idx in list(range(NUM_CHECKPOINTS)):
                pool.apply_async(
                    decompose_metrics,
                    args=("census", "vanilla", "tabnet", seed),
                    kwds={'checkpoint_idx': idx}
                )
        pool.close()
        pool.join()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for dataset in DATASETS:
            for method in METHODS:
                for classifier in MODELS:
                    for seed in RANDOM_SEED_LIST:
                        pool.apply_async(
                            calculate_compatibility, 
                            args=(dataset, method, classifier, seed)
                        )
        pool.close()
        pool.join()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for seed in RANDOM_SEED_LIST:
            for idx in list(range(NUM_CHECKPOINTS)):
                pool.apply_async(
                    calculate_compatibility,
                    args=("census", "vanilla", "tabnet", seed),
                    kwds={'checkpoint_idx': idx}
                )
        pool.close()
        pool.join()