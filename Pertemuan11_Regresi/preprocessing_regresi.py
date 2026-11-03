import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# TUGAS PERTEMUAN 11 - ANALISIS REGRESI KUALITAS AIR
# Kelompok 3 (Machine Learning)
# Bagian: Preprocessing & Data Transformation untuk Regresi
# Penanggung Jawab: Adel
# ==============================================================================

def load_and_preprocess_regression(scale_features=True, return_scaler=False):
    """
    Memuat dataset water_potability.csv dan menyiapkan data untuk analisis regresi.

    Transformasi Target & Fitur:
      1. Target (y) : Kolom 'ph' (data numerik kontinu).
      2. Fitur (X)  : 8 parameter kimia air (Hardness, Solids, Chloramines, 
                      Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity).
      3. Drop 'Potability': Dihapus karena merupakan label klasifikasi biner.
      
    Tahapan Preprocessing:
      1. Drop missing value pada target 'ph' (tidak diimputasi agar ground truth tetap valid).
      2. Imputasi missing values pada fitur X (Sulfate & Trihalomethanes) menggunakan Median.
      3. Deteksi dan penanganan outlier pada fitur X menggunakan IQR Clipping.
      4. Feature Scaling menggunakan StandardScaler (Z-score normalization).
      5. Split data menjadi 80% Training Set dan 20% Testing Set (random_state=42).

    Parameters:
      scale_features (bool): Apakah fitur X distandarisasi (StandardScaler). Default True.
      return_scaler (bool): Jika True, mengembalikan objek StandardScaler yang telah difit.

    Returns:
      X_train, X_test, y_train, y_test [, scaler]
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cek lokasi dataset di folder data/
    candidate_paths = [
        os.path.join(base_dir, "data", "water_potability.csv"),
        os.path.join(base_dir, "..", "data", "water_potability.csv"),
        os.path.join(os.getcwd(), "data", "water_potability.csv")
    ]
    
    data_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        raise FileNotFoundError("File water_potability.csv tidak ditemukan di direktori proyek.")

    df_raw = pd.read_csv(data_path)

    print("=" * 65)
    print(" PREPROCESSING REGRESI PREDIKSI NILAI pH AIR (KELOMPOK 3) ")
    print(" Penanggung Jawab: Adel")
    print("=" * 65)
    print(f"\n[1] Memuat Dataset Mentah:")
    print(f"    - Lokasi File  : {data_path}")
    print(f"    - Dimensi Awal : {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
    print(f"    - Missing Values Awal:")
    for col, null_count in df_raw.isnull().sum().items():
        if null_count > 0:
            print(f"      * {col:18s}: {null_count} data kosong ({null_count/len(df_raw)*100:.2f}%)")

    # ------------------------------------------------------------------
    # 1. PENANGANAN TARGET REGRESI ('ph')
    # ------------------------------------------------------------------
    # Nilai target kontinu tidak boleh diimputasi sintetis agar evaluasi performa model valid
    df = df_raw.dropna(subset=['ph']).copy()
    n_dropped_target = len(df_raw) - len(df)
    print(f"\n[2] Penanganan Kolom Target 'ph':")
    print(f"    - Menghapus {n_dropped_target} baris dengan nilai pH kosong.")
    print(f"    - Baris valid tersisa : {len(df)} sampel.")
    print(f"    - Statistik pH Aktual : Min={df['ph'].min():.2f}, Max={df['ph'].max():.2f}, Mean={df['ph'].mean():.2f}, Std={df['ph'].std():.2f}")

    # ------------------------------------------------------------------
    # 2. PEMISAHAN FITUR (X) DAN TARGET (y)
    # ------------------------------------------------------------------
    # Kolom 'Potability' di-drop karena merupakan target klasifikasi biner
    X = df.drop(columns=['ph', 'Potability'])
    y = df['ph']
    feature_names = list(X.columns)

    print(f"\n[3] Pemisahan Fitur dan Target:")
    print(f"    - Target (y) : 'ph' (Derajat Keasaman)")
    print(f"    - Fitur (X)  : {len(feature_names)} parameter kimia:")
    print(f"      {', '.join(feature_names)}")
    print(f"    - Kolom 'Potability' berhasil dihapus dari pemodelan regresi.")

    # ------------------------------------------------------------------
    # 3. SPLIT TRAIN & TEST (80% : 20%)
    # ------------------------------------------------------------------
    # Split dilakukan lebih awal untuk mencegah data leakage pada proses imputasi & scaling
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # 4. IMPUTASI MISSING VALUES PADA FITUR INPUT (X)
    # ------------------------------------------------------------------
    train_medians = X_train_raw.median(numeric_only=True)
    X_train_imp = X_train_raw.fillna(train_medians)
    X_test_imp = X_test_raw.fillna(train_medians)

    print(f"\n[4] Imputasi Missing Values pada Fitur X:")
    print(f"    - Metode Imputasi: Nilai Median dari Training Set (Mencegah Data Leakage)")
    for col in ['Sulfate', 'Trihalomethanes']:
        if col in train_medians:
            print(f"      * Median {col:16s}: {train_medians[col]:.4f}")
    print(f"    - Missing values tersisa di X_train: {X_train_imp.isnull().sum().sum()}")
    print(f"    - Missing values tersisa di X_test : {X_test_imp.isnull().sum().sum()}")

    # ------------------------------------------------------------------
    # 5. HANDLING OUTLIER DENGAN METODE IQR CLIPPING
    # ------------------------------------------------------------------
    outlier_counts = {}
    X_train_clipped = X_train_imp.copy()
    X_test_clipped = X_test_imp.copy()

    for col in feature_names:
        Q1 = X_train_imp[col].quantile(0.25)
        Q3 = X_train_imp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        n_outliers = ((X_train_imp[col] < lower_bound) | (X_train_imp[col] > upper_bound)).sum()
        outlier_counts[col] = n_outliers
        
        # Batasi nilai ekstrem ke batas kuartil (clipping)
        X_train_clipped[col] = X_train_clipped[col].clip(lower_bound, upper_bound)
        X_test_clipped[col] = X_test_clipped[col].clip(lower_bound, upper_bound)

    print(f"\n[5] Deteksi & Clipping Outlier (IQR Method 1.5):")
    for col, count in outlier_counts.items():
        print(f"    - {col:18s}: {count:3d} data pencilan di-clip")

    # ------------------------------------------------------------------
    # 6. FEATURE SCALING (STANDARISASI Z-SCORE)
    # ------------------------------------------------------------------
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clipped),
            columns=feature_names,
            index=X_train_clipped.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_clipped),
            columns=feature_names,
            index=X_test_clipped.index
        )
        print(f"\n[6] Feature Scaling:")
        print(f"    - Metode : StandardScaler (Mean=0, Std=1)")
        print(f"    - Skala fitur X_train berhasil distandarisasi.")
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        print(f"\n[6] Feature Scaling dilewati (Fitur dalam skala asli).")
        X_train_final = X_train_clipped
        X_test_final = X_test_clipped

    print(f"\n[7] Ringkasan Pembagian Data (Split 80:20):")
    print(f"    - Training Data (X_train, y_train): {X_train_final.shape[0]} baris x {X_train_final.shape[1]} fitur")
    print(f"    - Testing Data  (X_test, y_test)  : {X_test_final.shape[0]} baris x {X_test_final.shape[1]} fitur")
    print("=" * 65)
    print(" [SUKSES] Seluruh Tahapan Preprocessing Selesai Dijalankan! ")
    print("=" * 65 + "\n")

    if return_scaler:
        return X_train_final, X_test_final, y_train, y_test, scaler
    return X_train_final, X_test_final, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_regression(return_scaler=True)
    print("\nContoh 5 Baris Teratas X_train:")
    print(X_train.head())
    print("\nContoh 5 Baris Teratas y_train (Nilai pH):")
    print(y_train.head())

