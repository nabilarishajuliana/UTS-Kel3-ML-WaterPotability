import os
import sys
import matplotlib
matplotlib.use('Agg')  # Mode headless agar visualisasi tersimpan rapi tanpa popup

# Pastikan folder lokal ada di Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing_regresi import load_and_preprocess_regression
from linear_regression_pipeline import train_and_evaluate_linear_regression
from random_forest_pipeline import train_and_evaluate_random_forest
from gradient_boosting_pipeline import train_and_evaluate_gradient_boosting
from compare_regression import run_regression_comparison

# ==============================================================================
# MAIN RUNNER - TUGAS PERTEMUAN 11 REGRESI
# Kelompok 3: Water Potability (Adel, Juanne, Risha)
# ==============================================================================
# File ini menjalankan seluruh alur praktikum regresi secara berurutan:
# Step 1: Preprocessing & Data Splitting 80:20 (Adel)
# Step 2: Baseline Model - Linear Regression (Adel)
# Step 3: Model Juara - Random Forest Regressor (Juanne)
# Step 4: Model Pembanding - Gradient Boosting Regressor (Juanne)
# Step 5: Komparasi 5 Metrik & Ekspor Model untuk UI/UX Streamlit (Juanne & Risha)
# ==============================================================================

def main():
    print("#" * 70)
    print("  TUGAS PERTEMUAN 11: ANALISIS REGRESI KUALITAS AIR")
    print("  Kelompok 3: Prediksi Nilai pH (Adel, Juanne, Risha)")
    print("#" * 70)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing & Split Data 80:20 (Adel)
    # ------------------------------------------------------------------
    print("\n>>> STEP 1: PREPROCESSING & SPLIT DATA (Adel) <<<")
    X_train, X_test, y_train, y_test = load_and_preprocess_regression(scale_features=True)

    # ------------------------------------------------------------------
    # Step 2: Baseline Model - Linear Regression (Adel)
    # ------------------------------------------------------------------
    print("\n>>> STEP 2: BASELINE LINEAR REGRESSION PIPELINE (Adel) <<<")
    train_and_evaluate_linear_regression(save_plot=True)

    # ------------------------------------------------------------------
    # Step 3: Model Juara - Random Forest Regressor (Juanne)
    # ------------------------------------------------------------------
    print("\n>>> STEP 3: RANDOM FOREST REGRESSOR PIPELINE - MODEL JUARA (Juanne) <<<")
    train_and_evaluate_random_forest(save_plot=True, export_model=True)

    # ------------------------------------------------------------------
    # Step 4: Model Pembanding - Gradient Boosting (Juanne)
    # ------------------------------------------------------------------
    print("\n>>> STEP 4: GRADIENT BOOSTING REGRESSOR PIPELINE (Juanne) <<<")
    train_and_evaluate_gradient_boosting(save_plot=True, export_model=True)

    # ------------------------------------------------------------------
    # Step 5: Komparasi 3 Model 5 Metrik & Ekspor untuk UI/UX (Juanne & Risha)
    # ------------------------------------------------------------------
    print("\n>>> STEP 5: KOMPARASI MENYELURUH 3 MODEL & EKSPOR MODEL STREAMLIT <<<")
    df_results, best_model_name = run_regression_comparison(save_visualizations=True, export_best=True)

    print("\n" + "=" * 70)
    print("  SELESAI - Seluruh pipeline regresi berhasil dieksekusi 100%!")
    print(f"  Model Terbaik Terpilih : {best_model_name}")
    print("  Aplikasi Web Streamlit : Jalankan 'streamlit run app.py'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

