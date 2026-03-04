# ============================================================
# modules/data_loader.py — Loader Dataset IoT-23
# ============================================================

import pandas as pd
from io import StringIO
from config import DATA_RAW_PATH, TARGET_COLUMN, NORMAL_LABEL


def _parse_header(lines: list[str]) -> list[str]:
    """
    Ekstrak nama kolom dari baris '#fields' pada file conn.log.labeled.
    """
    for line in lines:
        if line.startswith("#fields"):
            columns = line.strip().split("\t")[1:]   # buang token '#fields'
            return columns
    raise ValueError("Header '#fields' tidak ditemukan dalam file.")


def load_raw(filepath: str = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Membaca file conn.log.labeled dari dataset IoT-23.

    Format file:
    - Baris komentar diawali '#'
    - Baris '#fields' berisi nama kolom (tab-separated)
    - Baris data dipisah oleh tab

    Returns
    -------
    pd.DataFrame : data mentah tanpa modifikasi apapun
    """
    print(f"[DataLoader] Membaca file: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    columns  = _parse_header(lines)
    data_str = "".join(l for l in lines if not l.startswith("#"))

    df = pd.read_csv(
        StringIO(data_str),
        sep="\t",
        names=columns,
        low_memory=False,
    )

    print(f"[DataLoader] ✅ Dataset dimuat — {df.shape[0]:,} baris, {df.shape[1]} kolom")
    return df


def summarize(df: pd.DataFrame) -> None:
    """
    Cetak ringkasan dataset: shape, tipe data, missing values,
    dan distribusi label.
    """
    print("\n" + "=" * 55)
    print("  RINGKASAN DATASET")
    print("=" * 55)
    print(f"  Shape      : {df.shape}")
    print(f"  Memori     : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing    : tidak ada")
    else:
        print(f"  Missing    :\n{missing}")

    if TARGET_COLUMN in df.columns:
        print(f"\n  Distribusi '{TARGET_COLUMN}':")
        counts = df[TARGET_COLUMN].value_counts()
        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"    {label:<35} {count:>8,}  ({pct:.2f}%)")

    print("=" * 55 + "\n")
