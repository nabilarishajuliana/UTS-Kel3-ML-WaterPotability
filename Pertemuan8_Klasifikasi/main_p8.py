import os
import sys
import matplotlib
matplotlib.use('Agg')  # mode headless agar visualisasi tersimpan rapi tanpa popup

# Pastikan folder lokal ada di Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing_p8 import load_and_preprocess
from decision_tree_pipeline import run_decision_tree_pipeline, visualize_decision_tree_result
from naive_bayes_pipeline import run_naive_bayes_pipeline, visualize_naive_bayes_result
from svm_pipeline import run_svm_pipeline, visualize_svm_result
from compare_models import build_comparison_table, compare_and_visualize


def main():
    print("#" * 70)
    print("  TUGAS PERTEMUAN 8: KLASIFIKASI NAIVE BAYES & SVM")
    print("  Kelompok 3: Water Potability (Adel, Juanne, Risha)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing & Split Data 80:20 (Adel)
    # ------------------------------------------------------------------
    print("\n>>> STEP 1: PREPROCESSING & SPLIT DATA (Adel) <<<")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # ------------------------------------------------------------------
    # Step 2: Baseline Decision Tree UTS SMOTE & RUS (Juanne & Risha)
    # ------------------------------------------------------------------
    print("\n>>> STEP 2: DECISION TREE PIPELINE — BASELINE UTS (Juanne & Risha) <<<")
    dt_res = run_decision_tree_pipeline(X_train, X_test, y_train, y_test)
    visualize_decision_tree_result(dt_res, X_train.columns)

    # ------------------------------------------------------------------
    # Step 3: Gaussian Naive Bayes SMOTE & RUS (Juanne)
    # ------------------------------------------------------------------
    print("\n>>> STEP 3: NAIVE BAYES PIPELINE (Juanne) <<<")
    nb_res = run_naive_bayes_pipeline(X_train, X_test, y_train, y_test)
    visualize_naive_bayes_result(nb_res)

    # ------------------------------------------------------------------
    # Step 4: Support Vector Machine (Kernel RBF) SMOTE & RUS (Risha)
    # ------------------------------------------------------------------
    print("\n>>> STEP 4: SUPPORT VECTOR MACHINE PIPELINE (Risha) <<<")
    svm_res = run_svm_pipeline(X_train, X_test, y_train, y_test)
    visualize_svm_result(svm_res)

    # ------------------------------------------------------------------
    # Step 5: Perbandingan Model 6 Konfigurasi & Visualisasi (Adel, Juanne, Risha)
    # ------------------------------------------------------------------
    print("\n>>> STEP 5: PERBANDINGAN METRIK 6 KONFIGURASI MODEL <<<")
    df_comp = build_comparison_table(dt_res, nb_res, svm_res)
    compare_and_visualize(df_comp)

    print("\n" + "=" * 70)
    print("  SELESAI - Seluruh pipeline berhasil dieksekusi 100%!")
    print("=" * 70)


if __name__ == "__main__":
    main()
