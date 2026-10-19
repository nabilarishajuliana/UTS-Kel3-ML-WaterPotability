import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
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
def run_decision_tree_pipeline(X_train, X_test, y_train, y_test):
    """
    Pipeline: Decision Tree (Baseline UTS) dengan evaluasi SMOTE & RUS.
    """
    print("=" * 60)
    print("DECISION TREE (Baseline UTS) — SMOTE & RUS")
    print("=" * 60)

    # 1. RESAMPLING
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    # SKENARIO A: DT + SMOTE (Sesuai tugas Juanne di UTS)
    print("\n" + "-" * 50)
    print("SKENARIO A: Decision Tree + SMOTE (Juanne - UTS)")
    print("-" * 50)
    dt_smote = DecisionTreeClassifier(random_state=42)
    dt_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = dt_smote.predict(X_test)
    m_smote = compute_metrics(y_test, y_pred_smote)

    print("Confusion Matrix (SMOTE):")
    print(f"  TN = {m_smote['tn']:4d} | FP = {m_smote['fp']:4d}")
    print(f"  FN = {m_smote['fn']:4d} | TP = {m_smote['tp']:4d}")
    print(f"  Accuracy    : {m_smote['accuracy']:.4f} ({m_smote['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m_smote['precision']:.4f} ({m_smote['precision']*100:.2f}%)")
    print(f"  Recall      : {m_smote['recall']:.4f} ({m_smote['recall']*100:.2f}%)")
    print(f"  Specificity : {m_smote['specificity']:.4f} ({m_smote['specificity']*100:.2f}%)")
    print(f"  F1-Score    : {m_smote['f1_score']:.4f} ({m_smote['f1_score']*100:.2f}%)")

    # SKENARIO B: DT + RUS (Sesuai tugas Risha di UTS)
    print("\n" + "-" * 50)
    print("SKENARIO B: Decision Tree + RUS (Risha - UTS)")
    print("-" * 50)
    dt_rus = DecisionTreeClassifier(random_state=42)
    dt_rus.fit(X_train_rus, y_train_rus)
    y_pred_rus = dt_rus.predict(X_test)
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
            'model': dt_smote,
            'y_pred': y_pred_smote,
            **m_smote
        },
        'rus': {
            'model': dt_rus,
            'y_pred': y_pred_rus,
            **m_rus
        }
    }


# ---------------------------------------------------------------
# FUNGSI VISUALISASI
# ---------------------------------------------------------------
def visualize_decision_tree_result(dt_results, feature_names):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Side-by-side Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Tidak Layak (0)', 'Layak (1)']

    sns.heatmap(dt_results['smote']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[0])
    acc_s = dt_results['smote']['accuracy'] * 100
    rec_s = dt_results['smote']['recall'] * 100
    axes[0].set_title(f"Confusion Matrix — Decision Tree + SMOTE\n(Akurasi: {acc_s:.2f}% | Recall: {rec_s:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')

    sns.heatmap(dt_results['rus']['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=axes[1])
    acc_r = dt_results['rus']['accuracy'] * 100
    rec_r = dt_results['rus']['recall'] * 100
    axes[1].set_title(f"Confusion Matrix — Decision Tree + RUS\n(Akurasi: {acc_r:.2f}% | Recall: {rec_r:.2f}%)",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'viz_dt_1_confusion_matrix.png'), dpi=120)
    plt.close()
    print("[Viz 1] Disimpan: viz_dt_1_confusion_matrix.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from preprocessing_p8 import load_and_preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess()
    res = run_decision_tree_pipeline(X_train, X_test, y_train, y_test)
    visualize_decision_tree_result(res, X_train.columns)
