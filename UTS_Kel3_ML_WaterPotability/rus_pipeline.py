import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main.py
# ---------------------------------------------------------------
def run_rus_pipeline(X_train, X_test, y_train, y_test):
    """
    Pipeline: RUS -> Decision Tree -> Evaluasi

    PENTING: RUS hanya diterapkan pada data TRAINING,
    data TESTING tidak boleh di-resample (biar merepresentasikan
    kondisi dunia nyata yang imbalanced).
    """
    print("=" * 50)
    print("RUS + DECISION TREE")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. RANDOM UNDER SAMPLING pada data training
    # ------------------------------------------------------------------
    print(f"\n[1] Distribusi y_train SEBELUM RUS:")
    print(f"    {dict(pd.Series(y_train).value_counts().sort_index())}")

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    print(f"\n    Distribusi y_train SESUDAH RUS:")
    print(f"    {dict(pd.Series(y_train_rus).value_counts().sort_index())}")
    print(f"    Shape X_train_rus : {X_train_rus.shape}")

    # ------------------------------------------------------------------
    # 2. TRAIN DECISION TREE pada data yang sudah di-RUS
    # ------------------------------------------------------------------
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train_rus, y_train_rus)

    print(f"\n[2] Model Decision Tree berhasil dilatih.")

    # ------------------------------------------------------------------
    # 3. PREDIKSI pada data testing (TIDAK di-resample)
    # ------------------------------------------------------------------
    y_pred = decision_tree.predict(X_test)
    print(f"\n[3] Prediksi selesai. Jumlah data test: {len(y_test)}")

    # ------------------------------------------------------------------
    # 4. EVALUASI: Confusion Matrix + Metrik
    # ------------------------------------------------------------------
    cm   = confusion_matrix(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n[4] Hasil Evaluasi:")
    print(f"    Confusion Matrix:")
    print(f"                    Predicted 0   Predicted 1")
    print(f"    Actual 0          {cm[0][0]:5d}         {cm[0][1]:5d}")
    print(f"    Actual 1          {cm[1][0]:5d}         {cm[1][1]:5d}")

    print(f"\n    Accuracy  : {acc:.4f}   ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}   ({prec*100:.2f}%)")
    print(f"    Recall    : {rec:.4f}   ({rec*100:.2f}%)")
    print(f"    F1-Score  : {f1:.4f}   ({f1*100:.2f}%)")

    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Tidak Layak (0)', 'Layak (1)']))
    print("[OK] Pipeline RUS selesai!\n")

    # Return semua hasil biar bisa dipakai buat visualisasi
    return {
        'model'           : decision_tree,
        'X_train_rus'     : X_train_rus,
        'y_train_rus'     : y_train_rus,
        'y_pred'          : y_pred,
        'confusion_matrix': cm,
        'accuracy'        : acc,
        'precision'       : prec,
        'recall'          : rec,
        'f1_score'        : f1
    }


# ---------------------------------------------------------------
# FUNGSI VISUALISASI — hanya dipanggil saat test mandiri
# ---------------------------------------------------------------
def visualize_rus_result(result, y_train, feature_names):
    """
    Menampilkan 4 visualisasi hasil pipeline RUS:
    1. Distribusi kelas y_train — Sebelum vs Sesudah RUS
    2. Confusion Matrix heatmap
    3. Bar chart metrik evaluasi (Accuracy, Precision, Recall, F1)
    4. Visualisasi Decision Tree
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ==============================================================
    # GAMBAR 1 — Distribusi Kelas Sebelum vs Sesudah RUS
    # ==============================================================
    y_before = pd.Series(y_train).value_counts().sort_index()
    y_after  = pd.Series(result['y_train_rus']).value_counts().sort_index()

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5))
    warna  = ['#E07A5F', '#3D405B']
    labels = ['Tidak Layak (0)', 'Layak (1)']

    bars_b = axes1[0].bar(labels, y_before.values, color=warna, width=0.5)
    axes1[0].set_title('SEBELUM RUS (Imbalanced)', fontweight='bold')
    axes1[0].set_ylabel('Jumlah Sampel')
    axes1[0].set_ylim(0, y_before.max() * 1.2)
    for bar, v in zip(bars_b, y_before.values):
        axes1[0].text(bar.get_x() + bar.get_width()/2, v + 20,
                      str(v), ha='center', fontweight='bold', fontsize=11)

    bars_a = axes1[1].bar(labels, y_after.values, color=warna, width=0.5)
    axes1[1].set_title('SESUDAH RUS (Balanced)', fontweight='bold')
    axes1[1].set_ylabel('Jumlah Sampel')
    axes1[1].set_ylim(0, y_before.max() * 1.2)
    for bar, v in zip(bars_a, y_after.values):
        axes1[1].text(bar.get_x() + bar.get_width()/2, v + 20,
                      str(v), ha='center', fontweight='bold', fontsize=11)

    fig1.suptitle('Distribusi Kelas y_train — Sebelum vs Sesudah RUS',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_rus_1_distribusi.png'),
                dpi=120, bbox_inches='tight')
    print("[Viz 1] Disimpan: viz_rus_1_distribusi.png")
    plt.show()

    # ==============================================================
    # GAMBAR 2 — Confusion Matrix Heatmap
    # ==============================================================
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False,
                xticklabels=['Tidak Layak (0)', 'Layak (1)'],
                yticklabels=['Tidak Layak (0)', 'Layak (1)'],
                annot_kws={'size': 16, 'fontweight': 'bold'}, ax=ax2)
    ax2.set_title('Confusion Matrix — RUS + Decision Tree',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Prediksi')
    ax2.set_ylabel('Aktual')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_rus_2_confusion_matrix.png'),
                dpi=120, bbox_inches='tight')
    print("[Viz 2] Disimpan: viz_rus_2_confusion_matrix.png")
    plt.show()

    # ==============================================================
    # GAMBAR 3 — Bar Chart Metrik Evaluasi
    # ==============================================================
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values  = [result['accuracy'], result['precision'],
               result['recall'], result['f1_score']]
    warna3  = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    bars = ax3.bar(metrics, values, color=warna3, width=0.55)
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel('Nilai (0 - 1)')
    ax3.set_title('Hasil Evaluasi RUS + Decision Tree',
                  fontsize=12, fontweight='bold')
    for bar, v in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f'{v:.4f}\n({v*100:.2f}%)', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_rus_3_metrik.png'),
                dpi=120, bbox_inches='tight')
    print("[Viz 3] Disimpan: viz_rus_3_metrik.png")
    plt.show()

    # ==============================================================
    # GAMBAR 4 — Visualisasi Decision Tree (dibatasi kedalamannya)
    # ==============================================================
    fig4, ax4 = plt.subplots(figsize=(20, 10))
    plot_tree(result['model'],
              feature_names=feature_names,
              class_names=['Tidak Layak', 'Layak'],
              filled=True, rounded=True,
              max_depth=3, fontsize=10, ax=ax4)
    ax4.set_title('Visualisasi Decision Tree (max_depth=3) — RUS',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_rus_4_decision_tree.png'),
                dpi=120, bbox_inches='tight')
    print("[Viz 4] Disimpan: viz_rus_4_decision_tree.png")
    plt.show()


# ---------------------------------------------------------------
# TEST MANDIRI — jalankan: python rus_pipeline.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing import load_and_preprocess

    # Ambil data hasil preprocessing (Adel)
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Jalankan pipeline RUS
    result = run_rus_pipeline(X_train, X_test, y_train, y_test)

    # Tampilkan semua visualisasi
    print("\n--- Membuat visualisasi... ---")
    visualize_rus_result(result, y_train,
                         feature_names=X_train.columns.tolist())
    print("\nSemua visualisasi selesai ditampilkan dan disimpan sebagai PNG.")