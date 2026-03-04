# ============================================================
# main.py — Entry Point Utama
# ============================================================
"""
Alur kerja:
  1. Load dataset IoT-23 (conn.log.labeled)
  2. Cleaning & preprocessing
  3. Feature selection + SMOTE
  4. Training semua model
  5. Evaluasi & perbandingan
  6. Simpan visualisasi & laporan
"""

import warnings
warnings.filterwarnings("ignore")

from modules.data_loader     import load_raw, summarize
from modules.preprocessor    import clean, split, scale
from modules.feature_engineer import select_features, get_feature_importance, apply_smote
from modules.trainer         import train_all, cross_validate_all
from modules.evaluator       import evaluate_all, get_confusion_matrix, save_report
from modules.visualizer      import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_feature_importance,
    plot_correlation_heatmap,
)
from config import DATA_RAW_PATH, TARGET_COLUMN


def main():
    print("\n" + "=" * 55)
    print("  IoT BOTNET CLASSIFIER — IoT-23 Dataset")
    print("=" * 55)

    # ── 1. Load Data ─────────────────────────────────────────
    df = load_raw(DATA_RAW_PATH)
    summarize(df)

    # ── 2. Visualisasi awal ──────────────────────────────────
    plot_label_distribution(df[TARGET_COLUMN])
    plot_correlation_heatmap(df)

    # ── 3. Cleaning & Split ──────────────────────────────────
    df_clean = clean(df)
    X_train, X_test, y_train, y_test = split(df_clean)

    # ── 4. Scaling ───────────────────────────────────────────
    feature_names = list(X_train.columns)
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)

    # ── 5. Feature Selection ─────────────────────────────────
    X_train_sel, X_test_sel, selected_names, selector = select_features(
        X_train_sc, y_train, X_test_sc, feature_names
    )
    df_importance = get_feature_importance(feature_names, selector)
    plot_feature_importance(df_importance)

    # ── 6. SMOTE ─────────────────────────────────────────────
    X_train_bal, y_train_bal = apply_smote(X_train_sel, y_train)

    # ── 7. Cross-Validation ──────────────────────────────────
    cross_validate_all(X_train_bal, y_train_bal)

    # ── 8. Training ──────────────────────────────────────────
    trained_models = train_all(X_train_bal, y_train_bal)

    # ── 9. Evaluasi ──────────────────────────────────────────
    df_results = evaluate_all(trained_models, X_test_sel, y_test)
    save_report(df_results)

    # ── 10. Visualisasi hasil ────────────────────────────────
    plot_model_comparison(df_results)
    plot_roc_curves(trained_models, X_test_sel, y_test)

    for name, model in trained_models.items():
        cm = get_confusion_matrix(model, X_test_sel, y_test)
        plot_confusion_matrix(cm, model_name=name)

    print("\n✅ Pipeline selesai! Cek folder outputs/ untuk laporan & grafik.\n")


if __name__ == "__main__":
    main()
