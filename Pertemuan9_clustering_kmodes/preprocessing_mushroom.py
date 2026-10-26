"""
preprocessing_mushroom.py

"""

import os
import pandas as pd


def load_and_preprocess_mushroom():
    """
    Load dataset mushrooms.csv, lakukan preprocessing lengkap,
    lalu return X (fitur) dan y (label untuk validasi hasil clustering).
    """

    # ------------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "mushrooms.csv")
    df = pd.read_csv(data_path)

    print("=" * 60)
    print("PREPROCESSING DATASET MUSHROOMS")
    print("=" * 60)
    print(f"\n[1] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"    Kolom pertama: {list(df.columns[:5])}...")

    # ------------------------------------------------------------------
    # 2. CEK DISTRIBUSI KELAS TARGET (untuk validasi nanti)
    # ------------------------------------------------------------------
    print(f"\n[2] Distribusi kelas 'class':")
    print(f"    {dict(df['class'].value_counts())}")
    print(f"    Keterangan: e = edible (bisa dimakan), p = poisonous (beracun)")

    # ------------------------------------------------------------------
    # 3. CEK MISSING VALUE FORMAL (NaN)
    # ------------------------------------------------------------------
    total_nan = df.isnull().sum().sum()
    print(f"\n[3] Missing values formal (NaN): {total_nan}")

    # ------------------------------------------------------------------
    # 4. HANDLE MISSING VALUE TERSEMBUNYI (tanda '?')
    # ------------------------------------------------------------------
    print(f"\n[4] Cek missing value tersembunyi (tanda '?'):")
    for col in df.columns:
        n_tanya = (df[col] == '?').sum()
        if n_tanya > 0:
            print(f"    Kolom '{col}': {n_tanya} data '?'")
            modus = df[df[col] != '?'][col].mode()[0]
            df[col] = df[col].replace('?', modus)
            print(f"      -> diganti dengan modus: '{modus}'")

    # ------------------------------------------------------------------
    # 5. DROP FITUR KONSTAN (nilai unik hanya 1)
    # ------------------------------------------------------------------
    print(f"\n[5] Cek fitur konstan (nilai unik = 1):")
    konstan = [col for col in df.columns if df[col].nunique() == 1]
    for col in konstan:
        print(f"    Kolom '{col}' konstan dengan nilai '{df[col].unique()[0]}'")
        print(f"      -> di-drop karena tidak informatif untuk clustering")
    df = df.drop(columns=konstan)

    # ------------------------------------------------------------------
    # 6. PISAHKAN FITUR (X) DAN LABEL VALIDASI (y)
    # ------------------------------------------------------------------
    X = df.drop(columns=['class'])  # 21 fitur kategorikal (setelah drop veil-type)
    y = df['class']  # label TRUE untuk validasi hasil cluster

    print(f"\n[6] Pemisahan Fitur dan Label:")
    print(f"    X (fitur) : {X.shape} -> {X.shape[1]} kolom kategorikal")
    print(f"    y (label) : {y.shape} -> label True hanya untuk validasi")
    print(f"\n    Catatan: Dalam clustering, label 'class' TIDAK digunakan untuk training.")
    print(f"    Label hanya dipakai di akhir untuk mengukur kualitas cluster.")

    print(f"\n[OK] Preprocessing selesai!\n")
    return X, y


# ---------------------------------------------------------------
# TEST MANDIRI - jalankan: python preprocessing_mushroom.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_and_preprocess_mushroom()
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    print(f"\nContoh 5 baris pertama X:")
    print(X.head())
    print(f"\nContoh 5 baris pertama y:")
    print(y.head())