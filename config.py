# ============================================================
# config.py — Konfigurasi Global Proyek
# ============================================================

import os

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_PATH   = os.path.join(BASE_DIR, "data", "raw", "conn.log.labeled")
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")
FIGURE_DIR      = os.path.join(BASE_DIR, "outputs", "figures")
REPORT_DIR      = os.path.join(BASE_DIR, "outputs", "reports")

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
TARGET_COLUMN   = "label"
NORMAL_LABEL    = "-"
DROP_COLUMNS    = ["ts", "uid", "id.orig_h", "id.resp_h", "tunnel_parents"]
CATEGORICAL_COLUMNS = ["proto", "service", "conn_state", "history"]

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------
TEST_SIZE    = 0.2
RANDOM_STATE = 42
SCALER       = "standard"      # "standard" | "minmax"

# ------------------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------------------
USE_FEATURE_SELECTION = True
N_FEATURES_TO_SELECT  = 15
FEATURE_METHOD        = "mutual_info"   # "mutual_info" | "chi2" | "rfe"

# ------------------------------------------------------------
# IMBALANCED DATA
# ------------------------------------------------------------
USE_SMOTE         = True
SMOTE_STRATEGY    = "auto"
SMOTE_K_NEIGHBORS = 5

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
MODELS_TO_TRAIN = [
    "random_forest",
    "decision_tree",
    "xgboost",
    "lightgbm",
    "logistic_regression",
]

MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "decision_tree": {
        "max_depth": 10,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}

# ------------------------------------------------------------
# EVALUASI
# ------------------------------------------------------------
CV_FOLDS     = 5
SCORING      = "f1_weighted"
AVERAGE_MODE = "weighted"

# ------------------------------------------------------------
# VISUALISASI
# ------------------------------------------------------------
FIGURE_SIZE   = (10, 7)
FIGURE_DPI    = 150
COLOR_PALETTE = "Set2"
