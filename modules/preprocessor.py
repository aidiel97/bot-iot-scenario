# ============================================================
# modules/preprocessor.py — Cleaning & Preprocessing
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from config import (
    TARGET_COLUMN, NORMAL_LABEL,
    DROP_COLUMNS, CATEGORICAL_COLUMNS,
    TEST_SIZE, RANDOM_STATE, SCALER,
)


# ── internal helpers ────────────────────────────────────────

def _drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[Preprocessor] Kolom dihapus : {cols_to_drop}")
    return df


def _replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Ganti placeholder Zeek ('-', '(empty)') dengan NaN."""
    df = df.replace({"-": np.nan, "(empty)": np.nan})
    return df


def _drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    null_ratio = df.isnull().mean()
    high_null  = null_ratio[null_ratio > threshold].index.tolist()
    df = df.drop(columns=high_null)
    if high_null:
        print(f"[Preprocessor] Kolom null >{threshold*100:.0f}% dihapus: {high_null}")
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binarisasi label:
      - Nilai NORMAL_LABEL  → 0 (benign)
      - Nilai lainnya       → 1 (botnet / malicious)
    """
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda x: 0 if str(x).strip() == NORMAL_LABEL else 1
    )
    benign  = (df[TARGET_COLUMN] == 0).sum()
    malicious = (df[TARGET_COLUMN] == 1).sum()
    print(f"[Preprocessor] Label — benign: {benign:,} | malicious: {malicious:,}")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")
            df[col] = le.fit_transform(df[col])
    print(f"[Preprocessor] Kolom kategorikal di-encode: {[c for c in CATEGORICAL_COLUMNS if c in df.columns]}")
    return df


def _convert_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Paksa kolom selain target menjadi numerik; yang gagal jadi NaN."""
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fill_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(0)
    return df


# ── public API ──────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline cleaning lengkap:
    drop → replace marker → drop high-null → encode target
    → encode categorical → convert numeric → fill NA
    """
    print("\n[Preprocessor] Mulai cleaning...")
    df = df.copy()
    df = _drop_irrelevant(df)
    df = _replace_missing_markers(df)
    df = _drop_high_null_columns(df)
    df = _encode_target(df)
    df = _encode_categoricals(df)
    df = _convert_numerics(df)
    df = _fill_na(df)
    print(f"[Preprocessor] ✅ Cleaning selesai — shape: {df.shape}")
    return df


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pisahkan fitur dan label, lalu train/test split stratified.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[Preprocessor] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Normalisasi fitur (fit pada train, transform pada keduanya).

    Returns
    -------
    X_train_scaled, X_test_scaled, fitted_scaler
    """
    scaler_cls = StandardScaler if SCALER == "standard" else MinMaxScaler
    scaler     = scaler_cls()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"[Preprocessor] Scaler '{SCALER}' diterapkan.")
    return X_train_scaled, X_test_scaled, scaler
