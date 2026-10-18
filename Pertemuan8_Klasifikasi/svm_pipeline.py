import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
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
def run_svm_pipeline(X_train, X_test, y_train, y_test):
    """
    Pipeline: Support Vector Machine (Kernel RBF) dengan evaluasi SMOTE & RUS.
    """
    print("=" * 60)
    print("SUPPORT VECTOR MACHINE (SVM - RBF Kernel) — SMOTE & RUS")
    print("=" * 60)

    # 1. RESAMPLING
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    print(f"\n[1] Persiapan Skenario Resampling:")
    print(f"    - SMOTE : {dict(pd.Series(y_train_smote).value_counts().sort_index())} (Total: {len(y_train_smote)})")
    print(f"    - RUS   : {dict(pd.Series(y_train_rus).value_counts().sort_index())} (Total: {len(y_train_rus)})")

    # ------------------------------------------------------------------
    # SKENARIO A: SVM + SMOTE
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("SKENARIO A: SVM (RBF) + SMOTE")
    print("-" * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_smote = SVC(kernel='rbf', probability=True, random_state=42)
        svm_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = svm_smote.predict(X_test)
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
    # SKENARIO B: SVM + RUS
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("SKENARIO B: SVM (RBF) + RUS")
    print("-" * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_rus = SVC(kernel='rbf', probability=True, random_state=42)
        svm_rus.fit(X_train_rus, y_train_rus)
    y_pred_rus = svm_rus.predict(X_test)
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
            'model': svm_smote,
            'y_pred': y_pred_smote,
            **m_smote
        },
        'rus': {
            'model': svm_rus,
            'y_pred': y_pred_rus,
            **m_rus
        }
    }


# ---------------------------------------------------------------
# FUNGSI VISUALISASI
# ---------------------------------------------------------------
def visualize_svm_result(svm_results):
    """
    Visualisasi hasil SVM:
    1. Side-by-side Confusion Matrix (SMOTE vs RUS)
    2. Bar chart perbandingan metrik evaluasi
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. SIDE-BY-SIDE CONFUSION MATRIX
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Tidak Layak (0)', 'Layak (1)']

    # Plot SMOTE
    sns.heatmap(svm_results['smote']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[0])
    acc_s = svm_results['smote']['accuracy'] * 100
    rec_s = svm_results['smote']['recall'] * 100
    axes[0].set_title(f"Confusion Matrix — SVM (RBF) + SMOTE\n(Akurasi: {acc_s:.2f}% | Recall: {rec_s:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')

    # Plot RUS
    sns.heatmap(svm_results['rus']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[1])
    acc_r = svm_results['rus']['accuracy'] * 100
    rec_r = svm_results['rus']['recall'] * 100
    axes[1].set_title(f"Confusion Matrix — SVM (RBF) + RUS\n(Akurasi: {acc_r:.2f}% | Recall: {rec_r:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_svm_1_confusion_matrix.png'), dpi=120)
    plt.close()
    print("[Viz 1] Disimpan: viz_svm_1_confusion_matrix.png")

    # 2. BAR CHART METRIK EVALUASI
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    smote_vals = [
        svm_results['smote']['accuracy'], svm_results['smote']['precision'],
        svm_results['smote']['recall'], svm_results['smote']['specificity'],
        svm_results['smote']['f1_score']
    ]
    rus_vals = [
        svm_results['rus']['accuracy'], svm_results['rus']['precision'],
        svm_results['rus']['recall'], svm_results['rus']['specificity'],
        svm_results['rus']['f1_score']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    b1 = ax.bar(x - width/2, smote_vals, width, label='SMOTE', color='#6366f1')
    b2 = ax.bar(x + width/2, rus_vals, width, label='RUS', color='#ec4899')

    ax.set_ylabel('Skor')
    ax.set_title('Perbandingan Metrik Evaluasi: SVM RBF (SMOTE vs RUS)', fontsize=12, fontweight='bold')
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
    plt.savefig(os.path.join(base_dir, 'viz_svm_2_metrik.png'), dpi=120)
    plt.close()
    print("[Viz 2] Disimpan: viz_svm_2_metrik.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from preprocessing_p8 import load_and_preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess()
    res = run_svm_pipeline(X_train, X_test, y_train, y_test)
    visualize_svm_result(res)
