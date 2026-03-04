# ============================================================
# modules/trainer.py — Training & Cross-Validation
# ============================================================

import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from config import (
    MODELS_TO_TRAIN, MODEL_PARAMS,
    MODEL_SAVE_DIR, CV_FOLDS, SCORING, RANDOM_STATE,
)


# ── Registry model ───────────────────────────────────────────

def _build_model(name: str):
    """Inisialisasi model berdasarkan nama string."""
    params = MODEL_PARAMS.get(name, {})

    registry = {
        "random_forest":      RandomForestClassifier,
        "decision_tree":      DecisionTreeClassifier,
        "logistic_regression": LogisticRegression,
    }

    if XGBClassifier:
        registry["xgboost"] = XGBClassifier
    if LGBMClassifier:
        registry["lightgbm"] = LGBMClassifier

    if name not in registry:
        raise ValueError(f"Model '{name}' tidak dikenal. Tersedia: {list(registry)}")

    return registry[name](**params)


# ── Training ─────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Latih semua model yang dikonfigurasi di MODELS_TO_TRAIN.

    Returns
    -------
    dict  { model_name: fitted_model }
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    trained = {}

    print(f"\n[Trainer] Melatih {len(MODELS_TO_TRAIN)} model...\n")

    for name in MODELS_TO_TRAIN:
        print(f"  ▶ {name}")
        model = _build_model(name)

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        trained[name] = model
        print(f"    ✅ Selesai dalam {elapsed:.2f}s")

        # Simpan model
        save_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"    💾 Disimpan ke {save_path}")

    print(f"\n[Trainer] ✅ Semua model selesai dilatih.")
    return trained


def cross_validate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Jalankan Stratified K-Fold Cross-Validation untuk semua model.

    Returns
    -------
    dict  { model_name: { "mean": float, "std": float, "scores": array } }
    """
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n[Trainer] Cross-Validation ({CV_FOLDS}-fold, scoring='{SCORING}')...\n")

    for name in MODELS_TO_TRAIN:
        model  = _build_model(name)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)

        results[name] = {
            "mean":   scores.mean(),
            "std":    scores.std(),
            "scores": scores,
        }
        print(f"  {name:<25} {SCORING}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def load_model(name: str):
    """Muat model yang sudah disimpan dari disk."""
    path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{name}' tidak ditemukan di {path}")
    model = joblib.load(path)
    print(f"[Trainer] Model '{name}' dimuat dari {path}")
    return model
