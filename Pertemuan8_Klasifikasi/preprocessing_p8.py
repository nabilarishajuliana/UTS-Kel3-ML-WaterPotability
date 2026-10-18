import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# FUNGSI PREPROCESSING — dipakai oleh semua pipeline di P8
# ---------------------------------------------------------------
def load_and_preprocess():
    """
    Load dataset water_potability.csv, lakukan preprocessing lengkap,
    lalu return X_train, X_test, y_train, y_test.

    Langkah:
      1. Load CSV
      2. Isi missing values dengan median
      3. Handle outlier dengan IQR clip
      4. Pisahkan fitur (X) dan target (y)
      5. Feature scaling: X / max
      6. Split 80% train / 20% test
    """

    # ------------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Data ada di folder parent (satu level di atas Pertemuan8_Klasifikasi)
    data_path = os.path.join(base_dir, "..", "data", "water_potability.csv")

    df = pd.read_csv(data_path)

    print("=" * 50)
    print("PREPROCESSING — Pertemuan 8 Klasifikasi")
    print("=" * 50)
    print(f"\n[1] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"    Kolom : {list(df.columns)}")
    print(f"\n    Missing values per kolom:\n{df.isnull().sum().to_string()}")

    # ------------------------------------------------------------------
    # 2. HANDLE MISSING VALUES — isi pakai median tiap kolom
    # ------------------------------------------------------------------
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"\n[2] Missing values setelah diisi median: {df.isnull().sum().sum()}")

    # ------------------------------------------------------------------
    # 3. DETEKSI & HANDLE OUTLIER — IQR method (clip ke batas bawah/atas)
    # ------------------------------------------------------------------
    outlier_summary = {}
    for col in df.select_dtypes(include=np.number).columns:
        if col == "Potability":       # jangan sentuh target
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_outlier
        df[col] = df[col].clip(lower, upper)

    print(f"\n[3] Outlier terdeteksi & di-clip per fitur:")
    for col, n in outlier_summary.items():
        print(f"    {col:25s}: {n} outlier")

    # ------------------------------------------------------------------
    # 4. PISAHKAN FITUR (X) DAN TARGET (y)
    # ------------------------------------------------------------------
    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    # ------------------------------------------------------------------
    # 5. FEATURE SCALING — bagi tiap fitur dengan nilai max-nya
    # ------------------------------------------------------------------
    X = X / X.max()
    print(f"\n[4] Feature Scaling selesai (X / max).")
    print(f"    Min global : {X.min().min():.4f}")
    print(f"    Max global : {X.max().max():.4f}")

    # ------------------------------------------------------------------
    # 6. SPLIT DATA 80% TRAIN / 20% TEST
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n[5] Split Data:")
    print(f"    X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"    X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"\n    Distribusi kelas y_train: {y_train.value_counts().to_dict()}")
    print(f"    Distribusi kelas y_test : {y_test.value_counts().to_dict()}")
    print("\n[OK] Preprocessing selesai!\n")

    return X_train, X_test, y_train, y_test


def get_resampled_data(X_train, y_train):
    """
    Menghasilkan data training yang telah di-resample dengan SMOTE dan RUS.
    Sesuai tahapan eksperimen UTS:
      - SMOTE: Oversampling kelas minoritas hingga seimbang 1:1
      - RUS  : Undersampling kelas mayoritas hingga seimbang 1:1
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    return (X_train_smote, y_train_smote), (X_train_rus, y_train_rus)


# ---------------------------------------------------------------
# TEST MANDIRI — jalankan: python preprocessing_p8.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("Shape X_train :", X_train.shape)
    print("Shape X_test  :", X_test.shape)
