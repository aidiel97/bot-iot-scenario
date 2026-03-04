# ============================================================
# modules/visualizer.py — Semua Fungsi Visualisasi
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

from config import FIGURE_DIR, FIGURE_SIZE, FIGURE_DPI, COLOR_PALETTE

os.makedirs(FIGURE_DIR, exist_ok=True)


def _save(filename: str) -> None:
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[Visualizer] 💾 Gambar disimpan: {path}")
    plt.close()


# ── Distribusi Label ─────────────────────────────────────────

def plot_label_distribution(y: pd.Series, title: str = "Distribusi Label") -> None:
    """Bar chart distribusi kelas (benign vs botnet)."""
    counts  = y.value_counts()
    labels  = ["Benign (0)", "Malicious (1)"]
    palette = sns.color_palette(COLOR_PALETTE, n_colors=len(counts))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(labels[:len(counts)], counts.values, color=palette)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=11)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Jumlah Sampel")
    plt.tight_layout()
    _save("label_distribution.png")


# ── Confusion Matrix ─────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: list[str] = ["Benign", "Malicious"],
) -> None:
    """Heatmap confusion matrix untuk satu model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(f"confusion_matrix_{model_name.replace(' ', '_')}.png")


# ── ROC Curve ────────────────────────────────────────────────

def plot_roc_curves(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Plot ROC curve semua model dalam satu grafik."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    palette = sns.color_palette(COLOR_PALETTE, n_colors=len(trained_models))

    for (name, model), color in zip(trained_models.items(), palette):
        try:
            y_prob   = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc  = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, lw=2)
        except AttributeError:
            print(f"[Visualizer] ⚠ {name} tidak mendukung predict_proba, dilewati.")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Semua Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save("roc_curves.png")


# ── Perbandingan Metrik ──────────────────────────────────────

def plot_model_comparison(df_results: pd.DataFrame) -> None:
    """
    Grouped bar chart perbandingan Accuracy, Precision, Recall, F1
    untuk semua model.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = df_results[["Model"] + metrics].set_index("Model")

    ax = df_plot.plot(
        kind="bar",
        figsize=(max(10, len(df_results) * 2), 6),
        colormap=COLOR_PALETTE,
        edgecolor="white",
        width=0.75,
    )
    ax.set_title("Perbandingan Metrik Antar Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    plt.tight_layout()
    _save("model_comparison.png")


# ── Feature Importance ───────────────────────────────────────

def plot_feature_importance(
    df_importance: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance / Score",
) -> None:
    """
    Horizontal bar chart skor fitur.
    df_importance harus memiliki kolom 'feature' dan 'score'.
    """
    if df_importance.empty or "score" not in df_importance.columns:
        print("[Visualizer] ⚠ df_importance kosong atau tidak memiliki kolom 'score'.")
        return

    df_top = df_importance.head(top_n).sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(df_top["feature"], df_top["score"],
                   color=sns.color_palette(COLOR_PALETTE, n_colors=len(df_top)))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Score")
    plt.tight_layout()
    _save("feature_importance.png")


# ── Correlation Heatmap ──────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 20) -> None:
    """Heatmap korelasi untuk N fitur numerik teratas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:top_n]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap (Top Fitur Numerik)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("correlation_heatmap.png")
