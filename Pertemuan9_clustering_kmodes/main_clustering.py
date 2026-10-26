"""
main_clustering.py

"""

import os
import pandas as pd
from preprocessing_mushroom import load_and_preprocess_mushroom
from kmodes_pipeline import (
    find_optimal_k,
    build_kmodes,
    evaluate_cluster,
    validate_with_true_class
)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "#" * 65)
    print("  TUGAS CLUSTERING - K-MODES untuk Dataset Mushrooms")
    print("  Kelompok 3")
    print("#" * 65)

    # ===================================================================
    # STEP 1: LOAD & PREPROCESSING
    # ===================================================================
    print("\n>>> STEP 1: Load & Preprocessing Data")
    X, y = load_and_preprocess_mushroom()

    # ===================================================================
    # STEP 2: PILIH FITUR UNTUK CLUSTERING
    # Alasan pakai subset (bukan semua 21 fitur):
    # - Clustering 21 dimensi sulit divisualisasikan
    # - Fitur dengan banyak varian cenderung paling informatif
    # - Dipilih 5 fitur yang menurut domain knowledge paling diskriminatif
    # ===================================================================
    print("\n>>> STEP 2: Memilih Subset Fitur untuk Clustering")
    FITUR_TERPILIH = [
        'odor',               # bau - paling prediktif untuk kelayakan jamur
        'cap-color',          # warna topi
        'habitat',            # lingkungan tumbuh
        'population',         # pola persebaran
        'spore-print-color'   # warna cetakan spora
    ]
    X_subset = X[FITUR_TERPILIH].copy()
    print(f"    Fitur dipilih: {FITUR_TERPILIH}")
    print(f"    Shape X_subset: {X_subset.shape}")

    # ===================================================================
    # STEP 3: ELBOW METHOD - Cari K Optimal
    # ===================================================================
    print("\n>>> STEP 3: Elbow Method")
    elbow_path = os.path.join(base_dir, "viz_elbow_kmodes.png")
    costs = find_optimal_k(X_subset, k_range=range(2, 11), save_path=elbow_path)

    # ===================================================================
    # STEP 4: BUILD K-MODES dengan K Terpilih
    # Pilih K berdasarkan hasil visual elbow method
    # Umumnya di dataset mushroom K optimal berada di sekitar 2-5
    # ===================================================================
    print("\n>>> STEP 4: Build K-Modes Final")
    K_OPTIMAL = 2  # Sesuaikan setelah lihat grafik elbow
    print(f"    K optimal yang dipilih: {K_OPTIMAL}")

    model, clusters = build_kmodes(X_subset, n_clusters=K_OPTIMAL)

    # ===================================================================
    # STEP 5: EVALUASI SILHOUETTE SCORE
    # ===================================================================
    print("\n>>> STEP 5: Evaluasi Silhouette Score")
    score = evaluate_cluster(X_subset, clusters)

    # ===================================================================
    # STEP 6: VALIDASI DENGAN LABEL TRUE
    # ===================================================================
    print("\n>>> STEP 6: Validasi dengan Label True")
    crosstab = validate_with_true_class(clusters, y)

    # ===================================================================
    # STEP 7: SIMPAN HASIL CLUSTERING KE CSV
    # ===================================================================
    print("\n>>> STEP 7: Simpan Hasil ke CSV")
    hasil = X_subset.copy()
    hasil['cluster'] = clusters
    hasil['true_class'] = y.values
    hasil_path = os.path.join(base_dir, "hasil_clustering_kmodes.csv")
    hasil.to_csv(hasil_path, index=False)
    print(f"    Hasil tersimpan: {hasil_path}")

    # ===================================================================
    # RINGKASAN AKHIR
    # ===================================================================
    print("\n" + "#" * 65)
    print("  RINGKASAN HASIL CLUSTERING")
    print("#" * 65)
    print(f"  Dataset         : {X.shape[0]} sampel x {X.shape[1]} fitur (full)")
    print(f"  Fitur dipakai   : {len(FITUR_TERPILIH)} fitur ({', '.join(FITUR_TERPILIH)})")
    print(f"  K optimal       : {K_OPTIMAL}")
    print(f"  Cost final      : {model.cost_:.2f}")
    print(f"  Silhouette Score: {score:.4f}")
    print(f"\n  File output     :")
    print(f"    - viz_elbow_kmodes.png")
    print(f"    - hasil_clustering_kmodes.csv")
    print("#" * 65 + "\n")