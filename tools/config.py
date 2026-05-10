NUM_SAMPLES = 500
NUM_CHECKPOINTS = 10
RANDOM_SEED_LIST = list(range(10))
DATASETS = ["census", "ufrgs", "compas", "diabetes", "bank", "heart"]
MODELS = ["lr", "svm", "dt", "rf", "mlp", "xgboost", "tabnet"]
METHODS = ["vanilla", "reweighing", "flipping", "blindness", "fairsmote", "maat"]

# train-test split ratio
SPLIT_RATIO = {
    "census": 0.3,
    "compas": 0.3,
    "ufrgs": 0.3,
    "diabetes": 0.3,
    "bank": 0.3,
    "heart": 0.3
}

# name and index of sensitive attributes
considered_sensitive_attributes = {
    "census": {"age": 0, "race": 6, "sex": 7},
    "compas": {"sex": 0, "age": 1, "race": 2},
    "ufrgs": {"sex": 0, "race": 1},
    "diabetes": {"age": 7},
    "bank": {"age": 0},
    "heart": {"age": 0, "sex": 1}
}

# privileged groups
privileged_groups = {
    "census": [{"age": 1}, {"race": 1}, {"sex": 1}],    # senior/white/male
    "compas": [{"sex": 1}, {"age": 2}, {"race": 1}],    # male/senior/Caucasian
    "ufrgs": [{"sex": 1}, {"race": 1}],         # male/white
    "diabetes": [{"age": 1}],               # young
    "bank": [{"age": 1}],     # young
    "heart": [{"age": 1}, {"sex": 1}]           # young/male
}

# unprivileged groups
unprivileged_groups = {
    "census": [{"age": 0}, {"race": 0}, {"sex": 0}],    # young/non-white/female
    "compas": [{"sex": 0}, {"age": 0}, {"race": 0}],    # female/young/non-Caucasian
    "ufrgs": [{"sex": 0}, {"race": 0}],         # female/non-white
    "diabetes": [{"age": 0}],               # senior
    "bank": [{"age": 0}],     # senior
    "heart": [{"age": 0}, {"sex": 0}]           # senior/female
}

# columns of preprocessed datasets
preprocessed_df_columns = {
    "census": [
        "age", "Workclass", "Education-Num", "Marital Status", "Occupation", "Relationship", "race",
        "sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Probability"
        ],
    "compas": [
            "sex", "age", "race", "decile_score", "priors_count", "c_charge_degree", "Probability"
        ],
    "ufrgs": [
            "sex", "race", "physics", "biology", "history", "second_language",
            "geography", "literature", "Portuguese_essay", "math",
            "chemistry", "Probability"
        ],
    "diabetes": [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","age","Probability"
    ],
    "bank": [
        "age","job","marital","education","default","balance","housing","loan","campaign","pdays",
        "previous","poutcome","Probability"
    ],
    "heart": [
        "age","sex","chest_pain_type","resting_blood_pressure","cholesterol","fasting_blood_sugar",
        "resting_electrocardiogram","max_heart_rate_achieved","exercise_induced_angina",
        "st_depression","st_slope","num_major_vessels","thalassemia","Probability"
    ]
}

# metrics decomposed
METRICS = ["accuracy", "recall", "FPR", "SPD", "EOD", "PED", "AOD", "CFVR", "GIFVR"]
list_group_fairness = ["SPD", "EOD", "PED", "AOD"]
list_individual_fairness = ["CFVR", "GIFVR"]
list_utilities = ["accuracy", "recall", "FPR"]

# colors/makers for figures
COLORS_PAIR = {
    'Group Fairness vs Utility': '#4477AA',
    'Individual Fairness vs Utility': '#EE6677',
    'Group Fairness vs Individual Fairness': '#228833',
    'Group Fairness vs Group Fairness': '#CCBB44',
    'Individual Fairness vs Individual Fairness': '#66CCEE',
    'Utility vs Utility': '#AA3377'
}
COLORS_METRIC = {
    'accuracy': '#117733',
    'recall': '#88CCEE',
    'FPR': '#44AA99',
    'SPD': '#332288',
    'EOD': '#882255',
    'PED': '#CC6677',
    'AOD': '#AA4499',
    'CFVR': '#DDCC77',
    'GIFVR': '#999933'
}
MARKERS_MODEL = {
    "lr": 'o',          # circle
    "svm": 's',         # square
    "dt": '^',          # triangle up
    "rf": 'D',          # diamond
    "mlp": 'v',         # triangle down
    "xgboost": 'p',     # pentagon
    "tabnet": '*'      # star
}

# full name of datasets
DATASETS_NAME = {
    "census": "Census Income",
    "ufrgs": "UFRGS",
    "compas": "COMPAS",
    "diabetes": "Diabetes",
    "bank": "Bank Marketing",
    "heart": "Heart Disease"
}

# full name of models
MODELS_NAME = {
    "lr": "LR",
    "svm": "SVM",
    "dt": "DT",
    "rf": "RF",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "tabnet": "TabNet"
}

# full name of methods
METHODS_NAME = {
    "vanilla": "Vanilla",
    "reweighing": "Reweighing",
    "flipping": "Flipping-based Training",
    "blindness": "Fairness through Blindness",
    "fairsmote": "Fair-SMOTE",
    "maat": "MAAT"
}

# Define the representative metric pairs to analyze
pairs_to_analyze = [
    ("accuracy", "AOD"),
    ("accuracy", "CFVR"),
    ("AOD", "CFVR")
]