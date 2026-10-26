"""
kmodes_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------
# 1. ELBOW METHOD - Cari jumlah cluster (K) optimal
# ---------------------------------------------------------------
def find_optimal_k(X, k_range=range(2, 11), save_path=None):
    """
    Jalankan K-Modes untuk beberapa nilai K, hitung cost (dissimilarity),
    lalu plot grafik Elbow untuk menentukan K optimal.

    Parameters:
        X        : DataFrame fitur kategorikal
        k_range  : rentang nilai K yang mau dicoba (default 2-10)
        save_path: path untuk simpan file PNG grafik elbow

    Returns:
        costs : list nilai cost untuk setiap K
    """
    print("\n" + "=" * 60)
    print("ELBOW METHOD - Mencari K Optimal")
    print("=" * 60)

    costs = []
    for k in k_range:
        print(f"  Training K-Modes dengan K={k}...", end=" ")
        km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=42)
        km.fit(X)
        costs.append(km.cost_)
        print(f"Cost = {km.cost_:.2f}")

    # Plot grafik Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), costs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Jumlah Cluster (K)', fontsize=12)
    plt.ylabel('Cost (Total Dissimilarity)', fontsize=12)
    plt.title('Elbow Method - K-Modes Clustering Mushrooms',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(k_range))

    # Highlight titik yang recommended (bisa manual lihat elbow)
    for i, (k, c) in enumerate(zip(k_range, costs)):
        plt.annotate(f'{c:.0f}', (k, c), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n[Viz] Grafik elbow disimpan: {save_path}")
    plt.show()

    return costs


# ---------------------------------------------------------------
# 2. BUILD K-MODES - Train model final
# ---------------------------------------------------------------
def build_kmodes(X, n_clusters, verbose=True):
    """
    Train K-Modes dengan K yang sudah dipilih dari Elbow Method.

    Parameters:
        X          : DataFrame fitur kategorikal
        n_clusters : jumlah cluster (K optimal)
        verbose    : True = tampilkan info detail

    Returns:
        model    : objek KModes yang sudah di-fit
        clusters : array label cluster per baris data (0 sampai K-1)
    """
    print("\n" + "=" * 60)
    print(f"BUILD K-MODES FINAL dengan K={n_clusters}")
    print("=" * 60)

    model = KModes(n_clusters=n_clusters, init='Huang', n_init=10,
                   verbose=0, random_state=42)
    clusters = model.fit_predict(X)

    print(f"\n[1] Model K-Modes berhasil dilatih.")
    print(f"    Final cost: {model.cost_:.2f}")
    print(f"    Jumlah iterasi: {model.n_iter_}")

    print(f"\n[2] Distribusi anggota per cluster:")
    for i in range(n_clusters):
        n_members = (clusters == i).sum()
        print(f"    Cluster {i}: {n_members} sampel ({n_members/len(clusters)*100:.1f}%)")

    if verbose:
        print(f"\n[3] Centroid tiap cluster (mode tiap fitur):")
        centroid_df = pd.DataFrame(model.cluster_centroids_, columns=X.columns)
        centroid_df.index = [f"Cluster {i}" for i in range(n_clusters)]
        print(centroid_df.T)

    return model, clusters


# ---------------------------------------------------------------
# 3. SILHOUETTE SCORE - Evaluasi kualitas cluster
# ---------------------------------------------------------------
def evaluate_cluster(X, clusters):
    """
    Hitung Silhouette Score untuk mengukur kualitas cluster.
    Karena data kategorikal, dilakukan encoding dulu lalu pakai metric 'hamming'.

    Rentang nilai: -1 (buruk) sampai +1 (sangat bagus).
    Interpretasi:
      > 0.5 : clustering sangat bagus
      0.25-0.5 : clustering cukup bagus
      < 0.25 : cluster tidak terdefinisi baik

    Parameters:
        X        : DataFrame fitur kategorikal
        clusters : array label cluster hasil K-Modes

    Returns:
        score : nilai Silhouette Score
    """
    print("\n" + "=" * 60)
    print("EVALUASI KUALITAS CLUSTER - Silhouette Score")
    print("=" * 60)

    # Encoding kategorikal ke numerik (hanya untuk perhitungan silhouette)
    X_encoded = X.apply(LabelEncoder().fit_transform)

    # Hitung Silhouette Score dengan metric Hamming (cocok untuk kategorikal)
    score = silhouette_score(X_encoded, clusters, metric='hamming')

    print(f"\n  Silhouette Score: {score:.4f}")

    if score > 0.5:
        kategori = "SANGAT BAGUS - cluster terpisah jelas"
    elif score > 0.25:
        kategori = "CUKUP BAGUS - cluster masih terpisah dengan reasonable"
    else:
        kategori = "KURANG BAGUS - cluster saling tumpang tindih"

    print(f"  Interpretasi: {kategori}")

    return score


# ---------------------------------------------------------------
# 4. VALIDASI CLUSTER DENGAN LABEL TRUE
# ---------------------------------------------------------------
def validate_with_true_class(clusters, y_true):
    """
    Bandingkan hasil cluster dengan label kelas sebenarnya (e vs p).
    Ini bukan untuk training, tapi untuk MEMVALIDASI apakah cluster
    yang terbentuk sesuai dengan pembagian edible vs poisonous.
    """
    print("\n" + "=" * 60)
    print("VALIDASI CLUSTER DENGAN LABEL TRUE")
    print("=" * 60)

    df_valid = pd.DataFrame({'cluster': clusters, 'true_class': y_true.values})
    crosstab = pd.crosstab(df_valid['cluster'], df_valid['true_class'])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    print(f"\n[1] Jumlah anggota (cluster vs true class):")
    print(crosstab)

    print(f"\n[2] Persentase per cluster:")
    print(crosstab_pct.round(2))

    print(f"\n[3] Dominasi tiap cluster:")
    for i in crosstab.index:
        majority = crosstab.loc[i].idxmax()
        pct = crosstab_pct.loc[i, majority]
        label = "Edible (bisa dimakan)" if majority == 'e' else "Poisonous (beracun)"
        print(f"    Cluster {i} didominasi oleh '{majority}' ({label}) sebanyak {pct:.1f}%")

    return crosstab