"""
This code is partially adapted from:
- https://github.com/LingfengZhang98/HIFI/blob/master/data/preprocess.py

Preprocess the original data, and save it to "{DATASET}_processed.csv".
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
script_dir = osp.dirname(osp.abspath(__file__))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, RANDOM_SEED_LIST, NUM_SAMPLES


def preprocess_census():
    DATASET = "census"
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        osp.join(script_dir, "tabular", DATASET, "raw_dataset", "adult.data"),
        names=[d[0] for d in dtypes],
        na_values=["?"],
        skipinitialspace=True,
        dtype=dict(dtypes)
    )
    raw_data = raw_data.dropna().reset_index(drop=True)  # drop data points that contains any N/A value
    dataset_orig = raw_data.drop(["Education", "fnlwgt"], axis=1)  # drop redundant or unrelated attributes
    print("The [Census Income] dataset is under preprocessing, and the mappings of sensitive attributes are:\n")
    print("If the 'age' >= 40, it is mapped to 1; otherwise, it is mapped to 0.\n")
    dataset_orig["Age"] = np.where(dataset_orig["Age"] >= 40, 1, 0)
    print("If the 'race' is 'White', it is mapped to 1; otherwise, it is mapped to 0.\n")
    print("If the 'sex' is 'Male', it is mapped to 1; otherwise, it is mapped to 0.\n")
    print("'income<=50K' gets 0, otherwise 1.\n\n")
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Education"]), dtypes))
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Race":
                dataset_orig[k] = np.where(dataset_orig[k] == "White", 1, 0)
            elif k == "Sex":
                dataset_orig[k] = np.where(dataset_orig[k] == "Male", 1, 0)
            elif k == "Target":
                dataset_orig[k] = np.where(dataset_orig[k] == "<=50K", 0, 1)
            elif k == "Country":
                dataset_orig[k] = np.where(dataset_orig[k] == "United-States", 1, 0)
            else:
                dataset_orig[k] = dataset_orig[k].cat.codes
    dataset_orig["Capital Gain"] = dataset_orig["Capital Gain"] // 10000
    dataset_orig["Capital Loss"] = dataset_orig["Capital Loss"] // 500
    dataset_orig["Hours per week"] = dataset_orig["Hours per week"] // 10
    dataset_orig.rename(index=str, columns={"Target": "Probability", "Age": "age", "Race": "race", "Sex": "sex"}, inplace=True)

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_ufrgs():
    DATASET = "ufrgs"
    columns_orig = ["gender", "race", "physics", "biology", "history", "second_language",
                    "geography", "literature", "Portuguese_essay", "math",
                    "chemistry", "mean_GPA"]
    dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", DATASET, "raw_dataset", "data_with_race.csv"), header=None, names=columns_orig)
    dataset_orig = dataset_orig.dropna()
    print("The [UFRGS] dataset is under preprocessing, and the mappings of sensitive attributes are:\n")
    print("For 'sex', 1 denotes male and 0 denotes female.\n")
    print("For 'race', 1 denotes 'White' and 0 denotes others.\n")
    dataset_orig["race"] = np.where(dataset_orig["race"] == 'White', 1, 0)
    print("'mean_GPA<3' gets 0, otherwise 1.\n\n")
    bins_score = [0, 300, 400, 500, 600, 700, 800, 900, 2000]
    for attribute in columns_orig:
        if attribute not in ["gender", "race", "mean_GPA"]:
            dataset_orig[attribute] = np.digitize(dataset_orig[attribute], bins_score)
    dataset_orig.rename(index=str, columns={"mean_GPA": "Probability", "gender": "sex"}, inplace=True)
    dataset_orig["Probability"] = np.where(dataset_orig["Probability"] < 3, 0, 1)

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_compas():
    DATASET = "compas"
    dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", DATASET, "raw_dataset", "compas-scores-two-years.csv"))
    dataset_orig = dataset_orig.drop(
        ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age_cat', 'juv_fel_count',
         'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
         'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
         'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
         'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
         'type_of_assessment', 'decile_score.1', 'priors_count.1', 'score_text', 'screening_date', 'v_type_of_assessment', 'v_decile_score',
         'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'event'], axis=1)
    dataset_orig = dataset_orig.dropna()
    print("The [COMPAS] dataset is under preprocessing, and the mappings of sensitive attributes are:\n")
    print("For 'sex', 1 denotes male and 0 denotes female.\n")
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 0, 1)
    print("For 'age', 0 denotes 'age < 25', 1 denotes '25<= age <= 45', and 2 denotes 'age > 45'.\n")
    dataset_orig['age'] = np.where(dataset_orig['age'] < 25, 0, np.where(dataset_orig["age"] > 45, 2, 1))
    print("For 'race', 1 denotes 'Caucasian' and 0 denotes others.\n")
    dataset_orig['race'] = np.where(dataset_orig['race'] == 'Caucasian', 1, 0)
    print("\n'recidivism' gets 0, otherwise 1.\n\n")
    bins_priors_count = [0, 1, 2, 3, 5, 7, 9, 11, 20, 30, 40]
    dataset_orig["priors_count"] = np.digitize(dataset_orig["priors_count"], bins_priors_count)
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
    dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_diabetes():
    DATASET = "diabetes"
    dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", DATASET, "raw_dataset", "diabetes.csv"))
    print("The [Diabetes] dataset is under preprocessing, and the mappings of sensitive attributes is:\n")
    print("If the 'age' < 30, it is mapped to 1; otherwise, it is mapped to 0.\n")
    dataset_orig["Age"] = np.where(dataset_orig["Age"] < 30, 1, 0)
    print("'diabetes' gets 0, otherwise 1.\n\n")
    bins_pregnancies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20]
    dataset_orig["Pregnancies"] = np.digitize(dataset_orig["Pregnancies"], bins_pregnancies)
    bins_glucose = [0, 50, 70, 90, 110, 130, 150, 170, 190, 210]
    dataset_orig["Glucose"] = np.digitize(dataset_orig["Glucose"], bins_glucose)
    bins_blood_pressure = [0, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    dataset_orig["BloodPressure"] = np.digitize(dataset_orig["BloodPressure"], bins_blood_pressure)
    bins_skin_thickness = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dataset_orig["SkinThickness"] = np.digitize(dataset_orig["SkinThickness"], bins_skin_thickness)
    bins_insulin = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    dataset_orig["Insulin"] = np.digitize(dataset_orig["Insulin"], bins_insulin)
    bins_bmi = [0, 20, 30, 40, 50, 60, 70]
    dataset_orig["BMI"] = np.digitize(dataset_orig["BMI"], bins_bmi)
    bins_diabetes_pedigree_function = np.arange(0, 2.6, 0.2)
    dataset_orig["DiabetesPedigreeFunction"] = np.digitize(dataset_orig["DiabetesPedigreeFunction"], bins_diabetes_pedigree_function)
    dataset_orig.rename(index=str, columns={"Outcome": "Probability", "Age": "age"}, inplace=True)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_bank():
    DATASET = "bank"
    dtypes = [
        ("age", "float32"), ("job", "category"), ("marital", "category"),
        ("education", "category"), ("default", "category"), ("balance", "float32"),
        ("housing", "category"), ("loan", "category"), ("contact", "category"),
        ("day", "float32"), ("month", "category"), ("duration", "float32"),
        ("campaign", "float32"), ("pdays", "float32"), ("previous", "float32"),
        ("poutcome", "category"), ("y", "category")
    ]
    dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", DATASET, "raw_dataset", "bank-full.csv"), sep=";", encoding='latin-1', dtype=dict(dtypes))

    # impute the missing values with the most frequent value
    dataset_orig[dataset_orig == 'unknown'] = np.nan
    for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
        dataset_orig[col].fillna(dataset_orig[col].mode()[0], inplace=True)

    dataset_orig = dataset_orig.drop(['contact', 'day', 'month', 'duration'], axis=1)

    categorical_variables = ["job", "marital", "education", "default", "housing", "loan", "poutcome", "y"]
    ecode = {
        "primary": 0,
        "secondary": 1,
        "tertiary": 2
    }
    for k in categorical_variables:
        if k == "Relationship":
            dataset_orig[k] = np.array([ecode[v.strip()] for v in dataset_orig[k]])
        elif k == "y":
            dataset_orig[k] = np.where(dataset_orig[k] == "no", 0, 1)
        else:
            dataset_orig[k] = dataset_orig[k].cat.codes

    for column in ["balance", "campaign", "pdays", "previous"]:
        column_data = dataset_orig[column]
        unique_values = np.unique(column_data)
        if len(unique_values) > 11:
            bins = np.linspace(min(unique_values), max(unique_values), num=11)
            new_values = np.digitize(column_data, bins) - 1
            new_values = np.clip(new_values, 0, 10)
        else:
            value_to_new = {v: i for i, v in enumerate(unique_values)}
            new_values = column_data.map(value_to_new)

        dataset_orig[column] = new_values

    dataset_orig.rename(index=str, columns={"y": "Probability"}, inplace=True)
    print("The [Bank Marketing] dataset is under preprocessing, and the mappings of sensitive attributes is:\n")
    print("If the 'age' < 40, it is mapped to 1; otherwise, it is mapped to 0.\n")
    dataset_orig["age"] = np.where(dataset_orig["age"] < 40, 1, 0)
    print("'no subscription' gets 0, otherwise 1.\n\n")

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_heart():
    DATASET = "heart"
    dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", DATASET, "raw_dataset", "heart.csv"))

    dataset_orig = dataset_orig[dataset_orig['ca'] < 4]  # drop the wrong ca values
    dataset_orig = dataset_orig[dataset_orig['thal'] > 0]  # drop the wong thal value

    dataset_orig = dataset_orig.rename(
        columns={'cp': 'chest_pain_type',
                 'trestbps': 'resting_blood_pressure',
                 'chol': 'cholesterol',
                 'fbs': 'fasting_blood_sugar',
                 'restecg': 'resting_electrocardiogram',
                 'thalach': 'max_heart_rate_achieved',
                 'exang': 'exercise_induced_angina',
                 'oldpeak': 'st_depression',
                 'slope': 'st_slope',
                 'ca': 'num_major_vessels',
                 'thal': 'thalassemia',
                 'target': 'Probability'},
        errors="raise")

    bins_rbp = np.linspace(dataset_orig["resting_blood_pressure"].min(), dataset_orig["resting_blood_pressure"].max(), 11)
    dataset_orig["resting_blood_pressure"] = np.digitize(dataset_orig["resting_blood_pressure"], bins_rbp, right=False)
    bins_chol = np.linspace(dataset_orig["cholesterol"].min(), dataset_orig["cholesterol"].max(), 11)
    dataset_orig["cholesterol"] = np.digitize(dataset_orig["cholesterol"], bins_chol, right=False)
    bins_mhra = np.linspace(dataset_orig["max_heart_rate_achieved"].min(), dataset_orig["max_heart_rate_achieved"].max(), 11)
    dataset_orig["max_heart_rate_achieved"] = np.digitize(dataset_orig["max_heart_rate_achieved"], bins_mhra, right=False)
    bins_sd = np.linspace(dataset_orig["st_depression"].min(), dataset_orig["st_depression"].max(), 11)
    dataset_orig["st_depression"] = np.digitize(dataset_orig["st_depression"], bins_sd, right=False)

    print("The [Heart Disease] dataset is under preprocessing, and the mappings of sensitive attributes is:\n")
    print("If the 'age' < 50, it is mapped to 1; otherwise, it is mapped to 0.\n")
    print("For 'sex', 1 denotes male and 0 denotes female.\n")
    dataset_orig["age"] = np.where(dataset_orig["age"] < 50, 1, 0) # mean age is approximately 54.5 years old
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)
    print("'normal' gets 1, otherwise 0.\n\n")

    dataset_orig.to_csv(osp.join(script_dir, "tabular", DATASET, DATASET+"_processed.csv"), index=False)


def majority_label_statistics(dataset_names):
    fout = open(osp.join(script_dir, "tabular_majority_label.txt"), "w")

    for dataset_name in dataset_names:
        fout.write(f"{dataset_name}\t")
    fout.write("\n")
    for dataset_name in dataset_names:
        dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", dataset_name, dataset_name + "_processed.csv"))
        label_avg = dataset_orig["Probability"].mean()
        if label_avg >= 0.5:
            fout.write("1(%.1f" % (100 * label_avg))
            fout.write("\\%)\t")
        else:
            fout.write("0(%.1f" % (100 * (1 - label_avg)))
            fout.write("\\%)\t")


def sample_test_data(test_data, num_samples=NUM_SAMPLES, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_size = len(test_data)
    
    if total_size <= num_samples:
        return test_data
    else:
        indices = np.random.choice(total_size, size=num_samples, replace=False)
        return test_data[indices]


def prepare_data(dataset_names, seed_list=RANDOM_SEED_LIST):
    for dataset_name in dataset_names:
        split_ratio = SPLIT_RATIO[dataset_name]
        dataset_orig = pd.read_csv(osp.join(script_dir, "tabular", dataset_name, dataset_name + "_processed.csv"))

        for seed in seed_list:
            set_seed(seed)

            data_save_root = osp.join(script_dir, "tabular", dataset_name, "prepared_data", "seed_" + str(seed))
            makedirs(data_save_root)

            data_train, data_test = train_test_split(dataset_orig.values, test_size=split_ratio, shuffle=True)
            np.save(osp.join(data_save_root, "data_train.npy"), data_train)
            np.save(osp.join(data_save_root, "data_test.npy"), data_test)
            
            data_test_sampled = sample_test_data(data_test)
            np.save(osp.join(data_save_root, "data_test_sampled.npy"), data_test_sampled)
            
            scaler = StandardScaler()
            scaler.fit(data_train[:, :-1])
            joblib.dump(scaler, osp.join(data_save_root, "scaler.pkl"))

            mean_baseline = np.mean(data_train[:, :-1], axis=0).reshape(1, -1)
            np.save(osp.join(data_save_root, "baseline.npy"), mean_baseline)

            constraints = np.vstack((data_train[:, :-1].min(axis=0), data_train[:, :-1].max(axis=0))).T.astype(np.int32)
            np.save(osp.join(data_save_root, "constraints.npy"), constraints)


if __name__ == '__main__':
    preprocess_census()
    preprocess_ufrgs()
    preprocess_compas()
    preprocess_diabetes()
    preprocess_bank()
    preprocess_heart()

    dataset_names = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
    majority_label_statistics(dataset_names)

    prepare_data(dataset_names)