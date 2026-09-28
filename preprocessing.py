import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main.py dan pipeline lain
# ---------------------------------------------------------------
def load_and_preprocess():
    """
    Load dataset water_potability.csv, lakukan preprocessing lengkap,
    lalu return X_train, X_test, y_train, y_test.
    """

    # ------------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "water_potability.csv")

    df = pd.read_csv(data_path)

    print("=" * 50)
    print("PREPROCESSING")
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
        if col == "Potability":          # jangan sentuh target
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_outlier
        df[col] = df[col].clip(lower, upper)   # clip -> replace outlier ke batas IQR

    print(f"\n[3] Outlier terdeteksi & di-clip per fitur:")
    for col, n in outlier_summary.items():
        print(f"    {col:25s}: {n} outlier")

    # ------------------------------------------------------------------
    # 4. PISAHKAN FITUR (X) DAN TARGET (y)
    # ------------------------------------------------------------------
    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    # ------------------------------------------------------------------
    # 5. SIMPLE FEATURE SCALING — bagi tiap fitur dengan nilai max-nya
    # ------------------------------------------------------------------
    X = X / X.max()

    print(f"\n[4] Feature Scaling selesai (X / max). Rentang tiap fitur:")
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


# ---------------------------------------------------------------
# FUNGSI VISUALISASI — hanya dipanggil saat test mandiri
# ---------------------------------------------------------------
def visualize_preprocessing():
    """
    Menampilkan visualisasi lengkap proses preprocessing:
    1. Missing Value bar chart
    2. Boxplot SEBELUM handle outlier
    3. Boxplot SESUDAH handle outlier (IQR clip)
    4. Distribusi kelas target (Potability)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "water_potability.csv")
    df_raw = pd.read_csv(data_path)

    fitur = [c for c in df_raw.columns if c != "Potability"]

    # ---- Simpan salinan data sebelum diproses ----
    df_before = df_raw.copy()
    df_before.fillna(df_before.median(numeric_only=True), inplace=True)
    # df_before: sudah isi median, BELUM di-clip (untuk boxplot "sebelum")

    # ---- Data sesudah clip ----
    df_after = df_before.copy()
    for col in fitur:
        Q1 = df_after[col].quantile(0.25)
        Q3 = df_after[col].quantile(0.75)
        IQR = Q3 - Q1
        df_after[col] = df_after[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # ==============================================================
    # GAMBAR 1 — Missing Value per Kolom
    # ==============================================================
    mv = df_raw.isnull().sum()
    mv = mv[mv > 0]   # hanya kolom yang ada missing value-nya

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    bars = ax1.bar(mv.index, mv.values, color=["#E07A5F", "#3D405B", "#81B29A"])
    ax1.set_title("Jumlah Missing Value per Kolom", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Kolom")
    ax1.set_ylabel("Jumlah Missing Value")
    ax1.set_ylim(0, mv.max() * 1.2)
    for bar, val in zip(bars, mv.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "viz_1_missing_value.png"), dpi=120)
    print("[Viz 1] Disimpan: viz_1_missing_value.png")
    plt.show()

    # ==============================================================
    # GAMBAR 2 — Boxplot SEBELUM Handle Outlier (IQR)
    # ==============================================================
    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10))
    axes2 = axes2.flatten()
    for i, col in enumerate(fitur):
        axes2[i].boxplot(df_before[col].dropna(), patch_artist=True,
                         boxprops=dict(facecolor="#F4A261", color="#264653"),
                         medianprops=dict(color="#E63946", linewidth=2),
                         whiskerprops=dict(color="#264653"),
                         capprops=dict(color="#264653"),
                         flierprops=dict(marker="o", markerfacecolor="#E63946",
                                         markersize=3, alpha=0.5))
        axes2[i].set_title(col, fontsize=9, fontweight="bold")
        axes2[i].set_xticks([])
    fig2.suptitle("Boxplot SEBELUM Handle Outlier (IQR)\n"
                  "(titik merah = outlier)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "viz_2_boxplot_sebelum.png"), dpi=120,
                bbox_inches="tight")
    print("[Viz 2] Disimpan: viz_2_boxplot_sebelum.png")
    plt.show()

    # ==============================================================
    # GAMBAR 3 — Boxplot SESUDAH Handle Outlier (IQR clip)
    # ==============================================================
    fig3, axes3 = plt.subplots(3, 3, figsize=(14, 10))
    axes3 = axes3.flatten()
    for i, col in enumerate(fitur):
        axes3[i].boxplot(df_after[col].dropna(), patch_artist=True,
                         boxprops=dict(facecolor="#2A9D8F", color="#264653"),
                         medianprops=dict(color="#E9C46A", linewidth=2),
                         whiskerprops=dict(color="#264653"),
                         capprops=dict(color="#264653"),
                         flierprops=dict(marker="o", markerfacecolor="#E76F51",
                                         markersize=3, alpha=0.5))
        axes3[i].set_title(col, fontsize=9, fontweight="bold")
        axes3[i].set_xticks([])
    fig3.suptitle("Boxplot SESUDAH Handle Outlier (IQR Clip)\n"
                  "(tidak ada titik outlier di luar whisker)", fontsize=13,
                  fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "viz_3_boxplot_sesudah.png"), dpi=120,
                bbox_inches="tight")
    print("[Viz 3] Disimpan: viz_3_boxplot_sesudah.png")
    plt.show()

    # ==============================================================
    # GAMBAR 4 — Distribusi Kelas Target (Potability)
    # ==============================================================
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    kelas = df_raw["Potability"].value_counts()
    warna = ["#E07A5F", "#3D405B"]
    bars4 = ax4.bar(["Tidak Layak (0)", "Layak Minum (1)"],
                    kelas.values, color=warna, width=0.5)
    ax4.set_title("Distribusi Kelas Target — Potability", fontsize=12,
                  fontweight="bold")
    ax4.set_ylabel("Jumlah Data")
    for bar, val in zip(bars4, kelas.values):
        pct = val / kelas.sum() * 100
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontweight="bold", fontsize=11)
    ax4.set_ylim(0, kelas.max() * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "viz_4_distribusi_kelas.png"), dpi=120)
    print("[Viz 4] Disimpan: viz_4_distribusi_kelas.png")
    plt.show()


# ---------------------------------------------------------------
# TEST MANDIRI — jalankan: python preprocessing.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Jalankan preprocessing dan tampilkan semua visualisasi
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("Shape X_train :", X_train.shape)
    print("Shape X_test  :", X_test.shape)
    print("Shape y_train :", y_train.shape)
    print("Shape y_test  :", y_test.shape)

    print("\n--- Membuat visualisasi... ---")
    visualize_preprocessing()
    print("\nSemua visualisasi selesai ditampilkan dan disimpan sebagai PNG.")