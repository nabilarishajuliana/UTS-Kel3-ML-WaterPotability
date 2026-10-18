import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ---------------------------------------------------------------
# HELPER: HITUNG METRIK
# ---------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'specificity': spec,
        'f1_score': f1
    }


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main_p8.py
# ---------------------------------------------------------------
def run_naive_bayes_pipeline(X_train, X_test, y_train, y_test):
    """
    Pipeline: Gaussian Naive Bayes dengan dua skenario resampling:
      - Skenario A: SMOTE (Oversampling)
      - Skenario B: RUS (Undersampling)
    """
    print("=" * 60)
    print("NAIVE BAYES (GaussianNB) — Evaluasi SMOTE & RUS")
    print("=" * 60)

    # 1. RESAMPLING
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    print(f"\n[1] Persiapan Skenario Resampling (Data Train Asli: {X_train.shape[0]} sampel):")
    print(f"    - SMOTE : {dict(pd.Series(y_train_smote).value_counts().sort_index())} (Total: {len(y_train_smote)})")
    print(f"    - RUS   : {dict(pd.Series(y_train_rus).value_counts().sort_index())} (Total: {len(y_train_rus)})")

    # ------------------------------------------------------------------
    # SKENARIO A: SMOTE
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("SKENARIO A: Naive Bayes + SMOTE")
    print("-" * 50)
    nb_smote = GaussianNB()
    nb_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = nb_smote.predict(X_test)
    m_smote = compute_metrics(y_test, y_pred_smote)

    print("Confusion Matrix (SMOTE):")
    print(f"  TN = {m_smote['tn']:4d} | FP = {m_smote['fp']:4d}")
    print(f"  FN = {m_smote['fn']:4d} | TP = {m_smote['tp']:4d}")
    print(f"  Accuracy    : {m_smote['accuracy']:.4f} ({m_smote['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m_smote['precision']:.4f} ({m_smote['precision']*100:.2f}%)")
    print(f"  Recall      : {m_smote['recall']:.4f} ({m_smote['recall']*100:.2f}%)")
    print(f"  Specificity : {m_smote['specificity']:.4f} ({m_smote['specificity']*100:.2f}%)")
    print(f"  F1-Score    : {m_smote['f1_score']:.4f} ({m_smote['f1_score']*100:.2f}%)")

    # ------------------------------------------------------------------
    # SKENARIO B: RUS
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("SKENARIO B: Naive Bayes + RUS")
    print("-" * 50)
    nb_rus = GaussianNB()
    nb_rus.fit(X_train_rus, y_train_rus)
    y_pred_rus = nb_rus.predict(X_test)
    m_rus = compute_metrics(y_test, y_pred_rus)

    print("Confusion Matrix (RUS):")
    print(f"  TN = {m_rus['tn']:4d} | FP = {m_rus['fp']:4d}")
    print(f"  FN = {m_rus['fn']:4d} | TP = {m_rus['tp']:4d}")
    print(f"  Accuracy    : {m_rus['accuracy']:.4f} ({m_rus['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m_rus['precision']:.4f} ({m_rus['precision']*100:.2f}%)")
    print(f"  Recall      : {m_rus['recall']:.4f} ({m_rus['recall']*100:.2f}%)")
    print(f"  Specificity : {m_rus['specificity']:.4f} ({m_rus['specificity']*100:.2f}%)")
    print(f"  F1-Score    : {m_rus['f1_score']:.4f} ({m_rus['f1_score']*100:.2f}%)")

    return {
        'smote': {
            'model': nb_smote,
            'y_pred': y_pred_smote,
            **m_smote
        },
        'rus': {
            'model': nb_rus,
            'y_pred': y_pred_rus,
            **m_rus
        }
    }


# ---------------------------------------------------------------
# FUNGSI VISUALISASI
# ---------------------------------------------------------------
def visualize_naive_bayes_result(nb_results):
    """
    Visualisasi hasil Naive Bayes:
    1. Side-by-side Confusion Matrix (SMOTE vs RUS)
    2. Bar chart perbandingan metrik evaluasi
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. SIDE-BY-SIDE CONFUSION MATRIX
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Tidak Layak (0)', 'Layak (1)']

    # Plot SMOTE
    sns.heatmap(nb_results['smote']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[0])
    acc_s = nb_results['smote']['accuracy'] * 100
    rec_s = nb_results['smote']['recall'] * 100
    axes[0].set_title(f"Confusion Matrix — Naive Bayes + SMOTE\n(Akurasi: {acc_s:.2f}% | Recall: {rec_s:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')

    # Plot RUS
    sns.heatmap(nb_results['rus']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[1])
    acc_r = nb_results['rus']['accuracy'] * 100
    rec_r = nb_results['rus']['recall'] * 100
    axes[1].set_title(f"Confusion Matrix — Naive Bayes + RUS\n(Akurasi: {acc_r:.2f}% | Recall: {rec_r:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_nb_1_confusion_matrix.png'), dpi=120)
    plt.close()
    print("[Viz 1] Disimpan: viz_nb_1_confusion_matrix.png")

    # 2. BAR CHART METRIK EVALUASI
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    smote_vals = [
        nb_results['smote']['accuracy'], nb_results['smote']['precision'],
        nb_results['smote']['recall'], nb_results['smote']['specificity'],
        nb_results['smote']['f1_score']
    ]
    rus_vals = [
        nb_results['rus']['accuracy'], nb_results['rus']['precision'],
        nb_results['rus']['recall'], nb_results['rus']['specificity'],
        nb_results['rus']['f1_score']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    b1 = ax.bar(x - width/2, smote_vals, width, label='SMOTE', color='#3b82f6')
    b2 = ax.bar(x + width/2, rus_vals, width, label='RUS', color='#10b981')

    ax.set_ylabel('Skor')
    ax.set_title('Perbandingan Metrik Evaluasi: Naive Bayes (SMOTE vs RUS)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_nb_2_metrik.png'), dpi=120)
    plt.close()
    print("[Viz 2] Disimpan: viz_nb_2_metrik.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from preprocessing_p8 import load_and_preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess()
    res = run_naive_bayes_pipeline(X_train, X_test, y_train, y_test)
    visualize_naive_bayes_result(res)
