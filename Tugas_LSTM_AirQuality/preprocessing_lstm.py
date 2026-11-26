import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main.py
# ---------------------------------------------------------------
def load_and_preprocess():
    """
    Load dataset AirQualityUCI.csv, lakukan pembersihan missing values,
    analisis korelasi, normalisasi fitur, dan sliding window,
    lalu return X_train, y_train, X_val, y_val, X_test, y_test, scaler.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "AirQualityUCI.csv")
    vis_dir = os.path.join(base_dir, "visualisasi")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 50)
    print("PREPROCESSING & TIME-SERIES WINDOWING (ADEL)")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, sep=';', decimal=',')
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

    # Parsing datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%d/%m/%Y %H.%M.%S', errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    print(f"\n[1] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"    Rentang waktu: {df['DateTime'].min()} s.d. {df['DateTime'].max()}")

    # ------------------------------------------------------------------
    # 2. HANDLE MISSING VALUES (-200) DENGAN LINEAR INTERPOLATION
    # ------------------------------------------------------------------
    feature_cols = [
        'C6H6(GT)',       # Target indeks 0: Konsentrasi Benzena (ug/m3)
        'PT08.S1(CO)',    # Sensor respons CO
        'PT08.S2(NMHC)',  # Sensor respons NMHC
        'PT08.S3(NOx)',   # Sensor respons NOx
        'PT08.S4(NO2)',   # Sensor respons NO2
        'PT08.S5(O3)',    # Sensor respons O3
        'T',              # Suhu udara (°C)
        'RH',             # Kelembaban relatif (%)
        'AH'              # Kelembaban mutlak
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(-200, np.nan)

    # Interpolasi linier khusus deret waktu
    df[feature_cols] = df[feature_cols].interpolate(method='linear').bfill().ffill()
    print(f"\n[2] Missing values (-200) setelah interpolasi linier: {df[feature_cols].isnull().sum().sum()}")

    # ------------------------------------------------------------------
    # 3. ANALISIS KORELASI & SIMPAN HEATMAP
    # ------------------------------------------------------------------
    corr_path = os.path.join(vis_dir, 'grafik_korelasi_fitur.png')
    plt.figure(figsize=(9, 7), dpi=300)
    sns.heatmap(df[feature_cols].corr(), annot=True, fmt='.2f', cmap='Blues', cbar=True,
                annot_kws={'size': 9, 'weight': 'bold'})
    plt.title('Matriks Korelasi Fitur Kualitas Udara (Air Quality UCI)\nPenanggung Jawab: Adella Putri Asy\'ari (434241023)',
              fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    print(f"\n[3] Heatmap korelasi disimpan ke: {corr_path}")

    # ------------------------------------------------------------------
    # 4. CHRONOLOGICAL SPLIT (70% TRAIN / 15% VAL / 15% TEST)
    # ------------------------------------------------------------------
    raw_values = df[feature_cols].values
    n_total = len(raw_values)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_raw = raw_values[:n_train]
    val_raw   = raw_values[n_train:n_train + n_val]
    test_raw  = raw_values[n_train + n_val:]

    print(f"\n[4] Pembagian data kronologis (no shuffle):")
    print(f"    Data Train : {len(train_raw)} baris ({len(train_raw)/n_total*100:.1f}%)")
    print(f"    Data Val   : {len(val_raw)} baris ({len(val_raw)/n_total*100:.1f}%)")
    print(f"    Data Test  : {len(test_raw)} baris ({len(test_raw)/n_total*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. FEATURE SCALING (MINMAXSCALER FIT PADA DATA LATIH)
    # ------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled   = scaler.transform(val_raw)
    test_scaled  = scaler.transform(test_raw)
    print(f"\n[5] Normalisasi MinMaxScaler selesai (rentang 0-1).")

    # ------------------------------------------------------------------
    # 6. PEMBENTUKAN SLIDING WINDOW 3D (TIMESTEPS = 24 JAM)
    # ------------------------------------------------------------------
    def _create_window(data, window_len=24):
        X, y = [], []
        for i in range(len(data) - window_len):
            X.append(data[i:i + window_len, :])
            y.append(data[i + window_len, 0])  # indeks 0 = C6H6(GT)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = _create_window(train_scaled, 24)
    X_val, y_val     = _create_window(val_scaled, 24)
    X_test, y_test   = _create_window(test_scaled, 24)

    print(f"\n[6] Pembentukan Tensor 3D Sliding Window selesai:")
    print(f"    Shape X_train : {X_train.shape}, y_train : {y_train.shape}")
    print(f"    Shape X_val   : {X_val.shape}, y_val   : {y_val.shape}")
    print(f"    Shape X_test  : {X_test.shape}, y_test  : {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
