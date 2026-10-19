import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_comparison_table(dt_res, nb_res, svm_res):
    """
    Menyusun tabel perbandingan lengkap untuk 6 konfigurasi model klasifikasi.
    """
    rows = [
        {
            'Model Klasifikasi': 'Decision Tree (UTS)',
            'Resampling': 'SMOTE',
            'Accuracy': dt_res['smote']['accuracy'],
            'Precision': dt_res['smote']['precision'],
            'Recall': dt_res['smote']['recall'],
            'Specificity': dt_res['smote']['specificity'],
            'F1-Score': dt_res['smote']['f1_score'],
            'TN': dt_res['smote']['tn'],
            'FP': dt_res['smote']['fp'],
            'FN': dt_res['smote']['fn'],
            'TP': dt_res['smote']['tp']
        },
        {
            'Model Klasifikasi': 'Decision Tree (UTS)',
            'Resampling': 'RUS',
            'Accuracy': dt_res['rus']['accuracy'],
            'Precision': dt_res['rus']['precision'],
            'Recall': dt_res['rus']['recall'],
            'Specificity': dt_res['rus']['specificity'],
            'F1-Score': dt_res['rus']['f1_score'],
            'TN': dt_res['rus']['tn'],
            'FP': dt_res['rus']['fp'],
            'FN': dt_res['rus']['fn'],
            'TP': dt_res['rus']['tp']
        },
        {
            'Model Klasifikasi': 'Naive Bayes (Gaussian)',
            'Resampling': 'SMOTE',
            'Accuracy': nb_res['smote']['accuracy'],
            'Precision': nb_res['smote']['precision'],
            'Recall': nb_res['smote']['recall'],
            'Specificity': nb_res['smote']['specificity'],
            'F1-Score': nb_res['smote']['f1_score'],
            'TN': nb_res['smote']['tn'],
            'FP': nb_res['smote']['fp'],
            'FN': nb_res['smote']['fn'],
            'TP': nb_res['smote']['tp']
        },
        {
            'Model Klasifikasi': 'Naive Bayes (Gaussian)',
            'Resampling': 'RUS',
            'Accuracy': nb_res['rus']['accuracy'],
            'Precision': nb_res['rus']['precision'],
            'Recall': nb_res['rus']['recall'],
            'Specificity': nb_res['rus']['specificity'],
            'F1-Score': nb_res['rus']['f1_score'],
            'TN': nb_res['rus']['tn'],
            'FP': nb_res['rus']['fp'],
            'FN': nb_res['rus']['fn'],
            'TP': nb_res['rus']['tp']
        },
        {
            'Model Klasifikasi': 'Support Vector Machine (RBF)',
            'Resampling': 'SMOTE',
            'Accuracy': svm_res['smote']['accuracy'],
            'Precision': svm_res['smote']['precision'],
            'Recall': svm_res['smote']['recall'],
            'Specificity': svm_res['smote']['specificity'],
            'F1-Score': svm_res['smote']['f1_score'],
            'TN': svm_res['smote']['tn'],
            'FP': svm_res['smote']['fp'],
            'FN': svm_res['smote']['fn'],
            'TP': svm_res['smote']['tp']
        },
        {
            'Model Klasifikasi': 'Support Vector Machine (RBF)',
            'Resampling': 'RUS',
            'Accuracy': svm_res['rus']['accuracy'],
            'Precision': svm_res['rus']['precision'],
            'Recall': svm_res['rus']['recall'],
            'Specificity': svm_res['rus']['specificity'],
            'F1-Score': svm_res['rus']['f1_score'],
            'TN': svm_res['rus']['tn'],
            'FP': svm_res['rus']['fp'],
            'FN': svm_res['rus']['fn'],
            'TP': svm_res['rus']['tp']
        }
    ]

    df_comp = pd.DataFrame(rows)
    return df_comp


def compare_and_visualize(df_comp):
    """
    Visualisasi perbandingan Accuracy dan F1-Score untuk seluruh konfigurasi
    (Persis seperti Gambar 6 pada modul laporan praktikum).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Cetak tabel perbandingan
    print("\n" + "=" * 90)
    print("TABEL PERBANDINGAN PERFORMA 6 KONFIGURASI MODEL KLASIFIKASI")
    print("=" * 90)

    display_df = df_comp.copy()
    display_df['Accuracy (%)'] = (display_df['Accuracy'] * 100).map('{:.2f}%'.format)
    display_df['Precision (%)'] = (display_df['Precision'] * 100).map('{:.2f}%'.format)
    display_df['Recall (%)'] = (display_df['Recall'] * 100).map('{:.2f}%'.format)
    display_df['Specificity (%)'] = (display_df['Specificity'] * 100).map('{:.2f}%'.format)
    display_df['F1-Score (%)'] = (display_df['F1-Score'] * 100).map('{:.2f}%'.format)

    cols_show = ['Model Klasifikasi', 'Resampling', 'Accuracy (%)', 'Precision (%)',
                 'Recall (%)', 'Specificity (%)', 'F1-Score (%)']
    print(display_df[cols_show].to_string(index=False))
    print("=" * 90)

    # Simpan CSV
    csv_path = os.path.join(base_dir, 'hasil_perbandingan_semua_model.csv')
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[OK] Disimpan file CSV: {csv_path}")

    # Buat Plot Grafik Batang Horizontal (Accuracy & F1-Score)
    labels = [f"{row['Model Klasifikasi'].replace('Support Vector Machine', 'SVM')}\n({row['Resampling']})"
              for _, row in df_comp.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Accuracy
    colors_acc = ['#60a5fa', '#3b82f6', '#f59e0b', '#d97706', '#10b981', '#059669']
    bars1 = axes[0].barh(labels[::-1], df_comp['Accuracy'].values[::-1], color=colors_acc[::-1], edgecolor='black', alpha=0.85)
    axes[0].set_title('Perbandingan Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars1:
        w = bar.get_width()
        axes[0].annotate(f'{w:.4f}', xy=(w, bar.get_y() + bar.get_height()/2),
                         xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

    # Subplot 2: F1-Score
    colors_f1 = ['#818cf8', '#6366f1', '#fbbf24', '#f59e0b', '#34d399', '#10b981']
    bars2 = axes[1].barh(labels[::-1], df_comp['F1-Score'].values[::-1], color=colors_f1[::-1], edgecolor='black', alpha=0.85)
    axes[1].set_title('Perbandingan F1-Score (Kelas Layak)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('F1-Score')
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars2:
        w = bar.get_width()
        axes[1].annotate(f'{w:.4f}', xy=(w, bar.get_y() + bar.get_height()/2),
                         xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    viz_path = os.path.join(base_dir, 'viz_compare_all_models.png')
    plt.savefig(viz_path, dpi=120)
    plt.close()
    print(f"[OK] Visualisasi disimpan: {viz_path}")

    # Cari Model Terbaik
    best_acc = df_comp.loc[df_comp['Accuracy'].idxmax()]
    best_f1 = df_comp.loc[df_comp['F1-Score'].idxmax()]

    print(f"\n[REKOMENDASI DEPLOYMENT]:")
    print(f"  - Model Akurasi Tertinggi : {best_acc['Model Klasifikasi']} + {best_acc['Resampling']} ({best_acc['Accuracy']*100:.2f}%)")
    print(f"  - Model F1-Score Tertinggi: {best_f1['Model Klasifikasi']} + {best_f1['Resampling']} ({best_f1['F1-Score']*100:.2f}%)")
    print("=" * 90)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from preprocessing_p8 import load_and_preprocess
    from naive_bayes_pipeline import run_naive_bayes_pipeline
    from svm_pipeline import run_svm_pipeline
    from decision_tree_pipeline import run_decision_tree_pipeline

    X_train, X_test, y_train, y_test = load_and_preprocess()
    dt_res = run_decision_tree_pipeline(X_train, X_test, y_train, y_test)
    nb_res = run_naive_bayes_pipeline(X_train, X_test, y_train, y_test)
    svm_res = run_svm_pipeline(X_train, X_test, y_train, y_test)

    df_comp = build_comparison_table(dt_res, nb_res, svm_res)
    compare_and_visualize(df_comp)
