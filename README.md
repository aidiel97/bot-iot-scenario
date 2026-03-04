# 🛡️ IoT Botnet Classifier — IoT-23 Dataset

Proyek klasifikasi traffic jaringan IoT untuk mendeteksi aktivitas botnet menggunakan dataset **IoT-23** (`conn.log.labeled`). Pipeline ini mencakup preprocessing, feature selection, penanganan class imbalance, training multi-model, evaluasi, dan visualisasi — semuanya terstruktur secara modular.

---

## 📁 Struktur Proyek

```
iot_botnet_classifier/
│
├── main.py                    # Entry point — jalankan pipeline penuh
├── config.py                  # Konfigurasi global (path, parameter, toggle)
├── requirements.txt
│
├── data/
│   └── raw/
│       └── conn.log.labeled   # ← Letakkan dataset di sini
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py         # Parser file IoT-23
│   ├── preprocessor.py        # Cleaning, encoding, normalisasi
│   ├── feature_engineer.py    # Feature selection & SMOTE
│   ├── trainer.py             # Training & cross-validation
│   ├── evaluator.py           # Metrik & laporan
│   └── visualizer.py          # Semua fungsi plotting
│
├── models/
│   └── saved/                 # Model tersimpan (.pkl)
│
└── outputs/
    ├── figures/               # Grafik hasil visualisasi (.png)
    └── reports/               # Laporan evaluasi (.csv)
```

---

## ⚙️ Instalasi

### 1. Clone / download proyek

```bash
git clone <repo-url>
cd iot_botnet_classifier
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

Dataset yang digunakan adalah **IoT-23** dari Stratosphere Laboratory.

- 🔗 Download: [https://www.stratosphereips.org/datasets-iot23](https://www.stratosphereips.org/datasets-iot23)
- File yang digunakan: `conn.log.labeled` (format Zeek/Bro log)
- Setelah download, letakkan file di:

```
data/raw/conn.log.labeled
```

> Path dapat dikonfigurasi ulang di `config.py` → `DATA_RAW_PATH`

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

Pipeline akan berjalan otomatis dari awal hingga akhir:

```
Load Data → Cleaning → Feature Selection → SMOTE → Training → Evaluasi → Simpan Output
```

---

## 🔄 Alur Pipeline

```
config.py
    │
    ▼
data_loader.py       ──▶  Baca & parse conn.log.labeled
    │
    ▼
preprocessor.py      ──▶  Drop kolom, encode label & kategorikal, scaling
    │
    ▼
feature_engineer.py  ──▶  Pilih N fitur terbaik, terapkan SMOTE
    │
    ▼
trainer.py           ──▶  Cross-validation + training, simpan model .pkl
    │
    ▼
evaluator.py         ──▶  Accuracy, Precision, Recall, F1, ROC-AUC
    │
    ▼
visualizer.py        ──▶  Confusion matrix, ROC curve, perbandingan model
```

---

## 🧩 Deskripsi Modul

| Modul | Tanggung Jawab |
|---|---|
| `config.py` | Satu tempat untuk mengubah semua parameter tanpa menyentuh kode utama |
| `data_loader.py` | Parsing header `#fields` khas format Zeek, ringkasan distribusi label |
| `preprocessor.py` | Drop kolom tak relevan, replace marker Zeek (`-`), LabelEncoder, StandardScaler |
| `feature_engineer.py` | SelectKBest (mutual info / chi2) atau RFE, oversampling SMOTE |
| `trainer.py` | Training loop multi-model, Stratified K-Fold CV, simpan/load `.pkl` |
| `evaluator.py` | Tabel perbandingan metrik semua model, export CSV |
| `visualizer.py` | 5 jenis plot: distribusi label, heatmap korelasi, ROC curve, confusion matrix, feature importance |

---

## ⚙️ Konfigurasi

Semua parameter dapat diubah di `config.py` tanpa menyentuh modul lain:

```python
# Contoh konfigurasi utama

MODELS_TO_TRAIN = ["random_forest", "xgboost", "lightgbm"]

USE_SMOTE             = True       # aktifkan/nonaktifkan SMOTE
USE_FEATURE_SELECTION = True       # aktifkan/nonaktifkan feature selection
N_FEATURES_TO_SELECT  = 15         # jumlah fitur terbaik
FEATURE_METHOD        = "mutual_info"  # "mutual_info" | "chi2" | "rfe"

SCALER       = "standard"          # "standard" | "minmax"
TEST_SIZE    = 0.2
CV_FOLDS     = 5
```

---

## 📊 Output yang Dihasilkan

### `outputs/figures/`

| File | Deskripsi |
|---|---|
| `label_distribution.png` | Distribusi kelas benign vs malicious |
| `correlation_heatmap.png` | Korelasi antar fitur numerik |
| `feature_importance.png` | Skor fitur dari metode seleksi |
| `roc_curves.png` | ROC curve semua model dalam satu grafik |
| `confusion_matrix_<model>.png` | Confusion matrix per model |
| `model_comparison.png` | Grouped bar chart perbandingan semua metrik |

### `outputs/reports/`

| File | Deskripsi |
|---|---|
| `classification_report.csv` | Tabel Accuracy, Precision, Recall, F1, ROC-AUC semua model |

### `models/saved/`

Model terbaik disimpan sebagai file `.pkl` dan dapat dimuat ulang:

```python
from modules.trainer import load_model

model = load_model("random_forest")
predictions = model.predict(X_test)
```

---

## 🤖 Model yang Didukung

| Nama di Config | Algoritma |
|---|---|
| `random_forest` | Random Forest Classifier |
| `decision_tree` | Decision Tree Classifier |
| `xgboost` | XGBoost Classifier |
| `lightgbm` | LightGBM Classifier |
| `logistic_regression` | Logistic Regression |

Tambah model baru cukup dengan mendaftarkannya di `config.py` → `MODELS_TO_TRAIN` dan `MODEL_PARAMS`, lalu daftarkan kelasnya di fungsi `_build_model()` dalam `trainer.py`.

---

## 📋 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## 📌 Catatan

- SMOTE **hanya** diterapkan pada data training, bukan data test.
- Label di-encode secara biner: `0` = benign, `1` = malicious (botnet).
- Kolom dengan nilai null lebih dari 60% akan otomatis dihapus.
- Semua placeholder Zeek (`-`, `(empty)`) diperlakukan sebagai `NaN`.

---

## 📚 Referensi

- **Dataset IoT-23**: Agustin Parmisano, Sebastian Garcia, Maria Jose Erquiaga. *A labeled dataset with malicious and benign IoT network traffic*. Stratosphere Laboratory, 2020.
- **Zeek Network Security Monitor**: [https://zeek.org](https://zeek.org)
- **imbalanced-learn (SMOTE)**: [https://imbalanced-learn.org](https://imbalanced-learn.org)