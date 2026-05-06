# ============================================================
# modules/feature_engineer.py — Feature Selection
# ============================================================

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

from config import (
    USE_FEATURE_SELECTION, N_FEATURES_TO_SELECT, FEATURE_METHOD,
    USE_SMOTE, SMOTE_STRATEGY, SMOTE_K_NEIGHBORS, RANDOM_STATE,
)


# ── Feature Selection ────────────────────────────────────────

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], object]:
    """
    Pilih N fitur terbaik menggunakan metode yang dikonfigurasi.

    Methods yang tersedia:
    - 'mutual_info' : Mutual Information (cocok untuk data campuran)
    - 'chi2'        : Chi-squared (hanya non-negatif)
    - 'rfe'         : Recursive Feature Elimination (berbasis Random Forest)

    Returns
    -------
    X_train_sel, X_test_sel, selected_feature_names, selector_object
    """
    if not USE_FEATURE_SELECTION:
        print("[FeatureEngineer] Feature selection dinonaktifkan.")
        return X_train, X_test, feature_names, None

    print(f"\n[FeatureEngineer] Metode: '{FEATURE_METHOD}' — memilih {N_FEATURES_TO_SELECT} fitur...")

    if FEATURE_METHOD == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=N_FEATURES_TO_SELECT)
        selector.fit(X_train, y_train)

    elif FEATURE_METHOD == "chi2":
        # chi2 butuh nilai non-negatif
        X_train_abs = np.abs(X_train)
        X_test_abs  = np.abs(X_test)
        selector = SelectKBest(score_func=chi2, k=N_FEATURES_TO_SELECT)
        selector.fit(X_train_abs, y_train)
        X_train = X_train_abs
        X_test  = X_test_abs

    elif FEATURE_METHOD == "rfe":
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        selector  = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=1)
        selector.fit(X_train, y_train)

    else:
        raise ValueError(f"FEATURE_METHOD tidak dikenal: '{FEATURE_METHOD}'")

    X_train_sel = selector.transform(X_train)
    X_test_sel  = selector.transform(X_test)

    # ambil nama fitur terpilih
    mask           = selector.get_support()
    selected_names = [name for name, selected in zip(feature_names, mask) if selected]

    print(f"[FeatureEngineer] ✅ Fitur terpilih ({len(selected_names)}):")
    for name in selected_names:
        print(f"   • {name}")

    return X_train_sel, X_test_sel, selected_names, selector


def get_feature_importance(
    feature_names: list[str],
    selector,
    method: str = FEATURE_METHOD,
) -> pd.DataFrame:
    """
    Kembalikan DataFrame skor kepentingan tiap fitur terpilih.
    Berguna untuk visualisasi feature importance.
    """
    if selector is None:
        return pd.DataFrame()

    if method in ("mutual_info", "chi2"):
        scores = selector.scores_
        mask   = selector.get_support()
        df = pd.DataFrame({
            "feature": feature_names,
            "score":   scores,
            "selected": mask,
        }).sort_values("score", ascending=False)

    elif method == "rfe":
        df = pd.DataFrame({
            "feature":  feature_names,
            "selected": selector.support_,
            "ranking":  selector.ranking_,
        }).sort_values("ranking")

    else:
        df = pd.DataFrame()

    return df
