# ============================================================
# modules/evaluator.py — Evaluasi Model
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)

from config import REPORT_DIR, AVERAGE_MODE


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Hitung semua metrik evaluasi untuk satu model.

    Returns
    -------
    dict berisi accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)

    # ROC-AUC (butuh probabilitas)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        roc_auc = None

    metrics = {
        "model":     model_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=AVERAGE_MODE, zero_division=0),
        "recall":    recall_score(y_test, y_pred, average=AVERAGE_MODE, zero_division=0),
        "f1":        f1_score(y_test, y_pred, average=AVERAGE_MODE, zero_division=0),
        "roc_auc":   roc_auc,
        "y_pred":    y_pred,
        "y_true":    y_test,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  EVALUASI: {model_name.upper()}")
        print(f"{'='*55}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    return metrics


def evaluate_all(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluasi semua model dan kembalikan DataFrame perbandingan.

    Parameters
    ----------
    trained_models : dict { model_name: fitted_model }

    Returns
    -------
    pd.DataFrame  — diurutkan berdasarkan F1-Score (descending)
    """
    all_results = []

    for name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, model_name=name)
        all_results.append({
            "Model":     result["model"],
            "Accuracy":  result["accuracy"],
            "Precision": result["precision"],
            "Recall":    result["recall"],
            "F1-Score":  result["f1"],
            "ROC-AUC":   result["roc_auc"],
        })

    df = pd.DataFrame(all_results).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print("\n[Evaluator] ✅ Perbandingan semua model:")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    return df


def get_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Kembalikan confusion matrix sebagai array numpy."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def save_report(df_results: pd.DataFrame, filename: str = "classification_report.csv") -> None:
    """Simpan DataFrame hasil evaluasi ke folder outputs/reports/."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    df_results.to_csv(path, index=False)
    print(f"[Evaluator] 💾 Laporan disimpan ke {path}")
