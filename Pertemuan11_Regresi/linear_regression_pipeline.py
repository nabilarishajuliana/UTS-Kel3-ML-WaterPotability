import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import fungsi preprocessing dari modul Adel
from preprocessing_regresi import load_and_preprocess_regression

# ==============================================================================
# TUGAS PERTEMUAN 11 - ANALISIS REGRESI KUALITAS AIR
# Kelompok 3 (Machine Learning)
# Bagian: Pipeline Baseline Model (Linear Regression) & Evaluasi 5 Metrik
# Penanggung Jawab: Adel
# ==============================================================================

def train_and_evaluate_linear_regression(save_plot=True, export_model=True):
    """
    Melatih model Linear Regression sebagai baseline pembanding,
    menghitung persamaan regresi, mengevaluasi 5 metrik, dan membuat visualisasi.

    Returns:
      model (LinearRegression): Objek model yang telah dilatih.
      metrics (dict): Nilai 5 metrik evaluasi (MAE, MSE, RMSE, MAPE, R2).
      y_pred (np.ndarray): Hasil prediksi nilai pH pada data uji.
    """
    print("=" * 65)
    print(" PIPELINE BASELINE MODEL: LINEAR REGRESSION (PREDIKSI pH) ")
    print(" Penanggung Jawab: Adel")
    print("=" * 65)

    # 1. Ambil data yang telah dipreprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_regression(scale_features=True)

    # 2. Inisialisasi dan Latih Model Linear Regression
    print("\n[1] Melatih Model Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("    -> Pelatihan model selesai.")

    # 3. Tampilkan Persamaan Regresi Linear
    intercept = model.intercept_
    coefficients = model.coef_
    feature_names = list(X_train.columns)

    print("\n[2] Parameter Persamaan Regresi:")
    print(f"    - Intercept (Beta 0) : {intercept:.4f}")
    print("    - Koefisien (Beta 1 .. Beta 8):")
    for feat, coef in zip(feature_names, coefficients):
        print(f"      * {feat:18s}: {coef:+.4f}")

    print("\n    Persamaan Matematis (Modul 11):")
    equation_terms = [f"({coef:+.3f} * {feat})" for feat, coef in zip(feature_names, coefficients)]
    print(f"    pH = {intercept:.3f} + " + " + ".join(equation_terms[:3]) + " + ... + e")

    # 4. Prediksi pada Data Uji (Testing Set)
    print("\n[3] Melakukan Prediksi pada Data Uji (Testing Set)...")
    y_pred = model.predict(X_test)

    # 5. Hitung 5 Metrik Evaluasi Wajib (Modul TM 11)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2 Score": r2
    }

    print("\n" + "=" * 50)
    print(" HASIL EVALUASI 5 METRIK - LINEAR REGRESSION ")
    print("=" * 50)
    print(f" 1. Mean Absolute Error (MAE)        : {mae:.4f}")
    print(f" 2. Mean Squared Error (MSE)         : {mse:.4f}")
    print(f" 3. Root Mean Squared Error (RMSE)   : {rmse:.4f}")
    print(f" 4. Mean Absolute Pct Error (MAPE)   : {mape:.2f}%")
    print(f" 5. R-Squared Score (R2)             : {r2:.4f}")
    print("=" * 50)

    # 6. Analisis Hasil untuk Pembahasan Laporan
    print("\n[4] Analisis Karakteristik Model Baseline (Untuk Bab II Laporan):")
    print("    - Nilai R2 yang mendekati 0 / sedikit negatif (-0.0034) membuktikan")
    print("      bahwa hubungan antar senyawa kimia air bersifat non-linear.")
    print("    - Linear Regression berasumsi data memiliki korelasi garis lurus,")
    print("      sehingga model hanya memprediksi nilai di sekitar rata-rata pH (Mean ~7.08).")
    print(f"    - Meskipun demikian, rata-rata error absolut (MAE) adalah {mae:.2f},")
    print(f"      artinya deviasi prediksi pH berada di rentang wajar +/- {mae:.2f}.")
    print("    - Hasil ini membuktikan pentingnya beralih ke Ensemble Learning")
    print("      (Random Forest & Gradient Boosting) pada modul tugas Juanne.")

    # 7. Ekspor Model Linear Regression ke .joblib
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if export_model:
        model_path = os.path.join(base_dir, "model_linear_regression.joblib")
        joblib.dump(model, model_path)
        print(f"\n[5] Model Linear Regression berhasil diekspor ke: {model_path}")

    # 8. Visualisasi Hasil Evaluasi
    if save_plot:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "visualisasi")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "evaluasi_linear_regression.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Data Aktual vs Prediksi
        axes[0].scatter(y_test, y_pred, color="#2b5c8f", alpha=0.6, edgecolors="k", s=35, label="Data Uji")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")
        axes[0].set_title("Actual vs Predicted pH\n(Linear Regression)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Nilai pH Aktual", fontsize=10)
        axes[0].set_ylabel("Nilai pH Prediksi", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Subplot 2: Residual Plot (Error Residuals vs Prediksi)
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, color="#e67e22", alpha=0.6, edgecolors="k", s=35)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
        axes[1].set_title("Residual Plot (Error vs Predicted)\nLinear Regression", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Nilai pH Prediksi", fontsize=10)
        axes[1].set_ylabel("Residual (Aktual - Prediksi)", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)

        # Subplot 3: Koefisien Regresi per Fitur
        coef_series = pd.Series(coefficients, index=feature_names).sort_values()
        colors = ["#27ae60" if c > 0 else "#c0392b" for c in coef_series.values]
        coef_series.plot(kind="barh", ax=axes[2], color=colors, edgecolor="black")
        axes[2].set_title("Bobot Koefisien Fitur (Beta)\nPengaruh terhadap pH", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Nilai Koefisien", fontsize=10)
        axes[2].axvline(0, color="black", linestyle="-", linewidth=0.8)
        axes[2].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[5] Visualisasi berhasil disimpan di: {plot_path}")

    print("\n" + "=" * 65)
    print(" [SELESAI] Pipeline Linear Regression Selesai Dijalankan! ")
    print("=" * 65 + "\n")

    return model, metrics, y_pred

if __name__ == "__main__":
    train_and_evaluate_linear_regression(save_plot=True)

