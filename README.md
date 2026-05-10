# A Game-Theoretic Framework for Measuring and Explaining Metric Compatibility in Fair Machine Learning

This repository contains the source code and supplementary materials for the paper **"A Game-Theoretic Framework for Measuring and Explaining Metric Compatibility in Fair Machine Learning"** (ICML 2026).

The code implements a unified framework to decompose fairness and utility metrics into attribute-level interactions using Harsanyi dividends, enabling the measurement of intrinsic compatibility between metrics.

## 📂 Project Structure

```text
.
├── data/                   # Dataset storage and preprocessing scripts
│   ├── preprocess.py       # Script to process raw data
│   └── [Raw Datasets]      # Raw CSV files are included here
├── environment/            # Environment configuration files
│   ├── environment.yml     # Conda environment file
│   └── requirements.txt    # Pip requirements file
├── exp/                    # Experiment scripts and results
│   ├── results/            # Output directory for figures and tables
│   ├── train_models.py     # Main script for model training
│   ├── calculate_compatibility.py # Core script for Harsanyi interaction calculation
│   └── [Analysis Scripts]  # Scripts to generate specific figures/tables
├── models/                 # Directory for saved models and interaction vectors
│   └── .keep               # Initially empty; populated during execution
├── tools/                  # Utility libraries
│   ├── config.py           # Global configuration
│   ├── decomposition.py    # Metric decomposition logic
│   └── ...
└── README.md
```

## 🛠️ Environment Setup

We provide environment files to easily reproduce the experimental setting. The code was tested on **Ubuntu 24.04.2 LTS** with **Python 3.10.18**.

### Option 1: Using Conda

```bash
cd {project_root}
conda env create -f environment/environment.yml
conda activate compatibility
```

### Option 2: Using Pip

```bash
cd {project_root}
pip install -r environment/requirements.txt
```

**Key Dependencies:**

* `numpy==2.1.2`, `pandas==2.3.3`, `scikit-learn==1.7.2`
* `torch==2.10.0`, `tensorflow==2.20.0`, `keras==3.11.3`
* `xgboost==3.0.1`, `pytorch-tabnet==4.1.0`
* `matplotlib==3.10.7`, `networkx==3.4.2`

## ⚙️ Hardware & Runtime Estimates

The experiments are designed to be computationally intensive due to the calculation of Harsanyi interactions.

* **CPU:** High-performance CPU recommended (e.g., AMD 9950X3D). The code supports multiprocessing based on available cores.
* **Memory:** Max usage is approx. **64GB** on a 16-core machine. Usage scales down with fewer cores.
* **GPU:** Not required. The experiments in the paper were conducted primarily on CPU.
* **Storage:** The full pipeline generates ~200GB of intermediate data (models, interaction vectors) in the `models/` directory. Ensure sufficient disk space.
* **Runtime:** A full reproduction (training + interaction calculation) takes approximately **3-5 days** on a high-end workstation (16+ cores).

## 🚀 Reproduction Instructions

To reproduce all results and figures presented in the paper, please execute the scripts in the following order.

### 1. Data Preprocessing

Prepare the 6 datasets (Census Income, UFRGS, COMPAS, Diabetes, Bank Marketing, Heart Disease).

```bash
python data/preprocess.py
```

### 2. Model Training (Time Consuming)

Train 7 types of ML models across 6 datasets using Vanilla training and 5 debiasing methods (10 random seeds each).

```bash
# This script will populate the models/ directory
python exp/train_models.py
```

### 3. Basic Evaluation

Evaluate the performance of trained models on 9 metrics (Original & Approximate).

```bash
# Reproduces Table 9 in the paper
python exp/evaluate_methods.py
```

### 4. Proxy Validation

Verify the correlation between original discrete metrics and our continuous proxies.

```bash
# Reproduces Table 4 in the paper
python exp/test_proxy.py
```

### 5. Compatibility Calculation (Time Consuming)

Compute Harsanyi interactions, decompose metrics, and calculate pairwise compatibility.

```bash
# This script will populate the models/ directory
python exp/calculate_compatibility.py
```

### 6. Figure & Table Generation

Once step 5 is complete, run the following scripts to generate the specific artifacts found in the paper. All outputs are saved in `exp/results/`.

| Script Name                                | Paper Artifact        | Description                                                |
|:------------------------------------------ |:--------------------- |:---------------------------------------------------------- |
| `python exp/plot4RQ1_landscape.py`         | **Figure 1**          | Global compatibility landscape boxplots.                   |
| `python exp/plot4RQ2_PCA.py`               | **Figure 4**          | PCA visualization of metric interaction vectors.           |
| `python exp/plot4RQ2_order.py`             | **Figure 2**          | Violin plots of coalition contributions by order.          |
| `python exp/plot4RQ2_example.py`           | **Tables 6, 10-12**   | Top explanatory coalitions for specific pairs.             |
| `python exp/plot4RQ3_training_dynamics.py` | **Figure 6**          | Evolution of compatibility during TabNet training.         |
| `python exp/plot4RQ4_debiasing.py`         | **Figure 5, Table 7** | Compatibility networks and average scores under debiasing. |
| `python exp/plot4RQ5_predictability.py`    | **Figure 7, Table 8** | ANOVA analysis and predictability experiments.             |

### 7. Case Study (HIFI)

Run the specific case study on HIFI regularization dynamics.

```bash
# Reproduces Figure 3 in the paper
python exp/explore_compatibility_with_HIFI.py
```

## 🧩 Extensibility

This framework is designed to be modular. You can extend it for your own research:

* **New Datasets:** Add the CSV file to `data/` and implement a preprocessing function in `data/preprocess.py`.
* **New Metrics:** Implement the decomposition logic (must be expressible as linear expectations) in `tools/decomposition.py` and register it in `tools/evaluation.py`.
* **New Models:** Add the model class wrapper in `tools/models.py`.
* **New Debiasing Methods:** Implement the training logic in `exp/train_models.py`.
* **Configuration:** Modify `tools/config.py` to change hyperparameters, seed counts, or sampling sizes.

## ⚠️ Notes

* **Intermediate Files:** The `models/` directory is excluded from the submission due to size constraints (>200GB). It will be automatically created and populated when running `exp/train_models.py` and `exp/calculate_compatibility.py`.

## License

MIT License.
