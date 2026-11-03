import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_regresi import load_and_preprocess_regression

# ==============================================================================
# TUGAS PERTEMUAN 11 - ANALISIS REGRESI KUALITAS AIR
# Kelompok 3 (Machine Learning)
# Bagian: Komparasi Menyeluruh 3 Model Regresi & Penentuan Model Terbaik
# Penanggung Jawab: Juanne
# ==============================================================================

def run_regression_comparison(save_visualizations=True, export_best=True):
    """
    Melatih 3 model regresi secara head-to-head, membandingkan 5 metrik evaluasi,
    menyimpan rekap perbandingan ke CSV, dan menghasilkan visualisasi komparatif.
    """
    print("=" * 70)
    print(" KOMPARASI MENYELURUH 3 MODEL REGRESI PREDIKSI pH (KELOMPOK 3) ")
    print(" Penanggung Jawab: Juanne")
    print("=" * 70)

    # 1. Load data & ambil scaler
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_regression(
        scale_features=True, return_scaler=True
    )

    # 2. Definisikan 3 Model Regresi
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "Random Forest Regressor (Juara)": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1
        )
    }

    results = []
    predictions = {}
    trained_models = {}

    print("\n[1] Memulai Pelatihan dan Evaluasi Seluruh Model...")
    for name, model in models.items():
        print(f"    - Melatih: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        predictions[name] = y_pred
        trained_models[name] = model

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAPE (%)": round(mape, 2),
            "R2 Score": round(r2, 4)
        })

    # 3. Tampilkan Tabel Komparasi
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" TABEL KOMPARASI PERFORMA MODEL (5 METRIK EVALUASI) ")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)

    # 4. Simpan Tabel Rekap ke File CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "hasil_komparasi_regresi.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[2] Tabel komparasi disimpan ke CSV: {csv_path}")

    # 5. Tentukan Model Terbaik (Juara)
    # Model terbaik dipilih berdasarkan RMSE terendah & R2 tertinggi
    best_row = df_results.sort_values(by=["RMSE", "R2 Score"], ascending=[True, False]).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    print("\n[3] Penentuan Model Terbaik:")
    print(f"    [JUARA] Model Terbaik : {best_model_name}")
    print(f"    - Skor RMSE      : {best_row['RMSE']} (Terendah)")
    print(f"    - Skor R2        : {best_row['R2 Score']} (Tertinggi)")
    print(f"    - Skor MAE       : {best_row['MAE']} (Deviasi ~{best_row['MAE']:.2f} pH)")

    # 6. Ekspor Model Terbaik, Model Baseline, dan Scaler untuk Streamlit
    if export_best:
        best_model_path = os.path.join(base_dir, "model_terbaik.joblib")
        lr_model_path = os.path.join(base_dir, "model_linear_regression.joblib")
        scaler_path = os.path.join(base_dir, "scaler.joblib")
        joblib.dump(best_model, best_model_path)
        joblib.dump(trained_models["Linear Regression (Baseline)"], lr_model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\n[4] File untuk UI/UX Streamlit berhasil diekspor:")
        print(f"    - Model Terbaik (RF): {best_model_path}")
        print(f"    - Model Baseline (LR): {lr_model_path}")
        print(f"    - Scaler Fitur      : {scaler_path}")

    # 7. Visualisasi Komparatif
    if save_visualizations:
        output_dir = os.path.join(base_dir, "visualisasi")
        os.makedirs(output_dir, exist_ok=True)
        comp_plot_path = os.path.join(output_dir, "komparasi_model_regresi.png")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        model_labels = ["Linear Reg", "Gradient Boost", "Random Forest"]
        colors = ["#3498db", "#9b59b6", "#2ecc71"]

        # Panel 1: Perbandingan MAE & RMSE
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(model_labels))
        width = 0.35
        ax1.bar(x - width/2, df_results["MAE"], width, label="MAE", color="#34495e", edgecolor="black")
        ax1.bar(x + width/2, df_results["RMSE"], width, label="RMSE", color="#e74c3c", edgecolor="black")
        ax1.set_title("Perbandingan Error (MAE vs RMSE)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, fontsize=10)
        ax1.set_ylabel("Nilai Error (Makin kecil makin baik)", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Perbandingan R2 Score
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(model_labels, df_results["R2 Score"], color=colors, edgecolor="black", width=0.5)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_title("Perbandingan R² Score (Makin tinggi makin baik)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Skor R²", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        for bar in bars2:
            yval = bar.get_height()
            offset = 0.001 if yval >= 0 else -0.003
            ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Panel 3: Perbandingan MAPE (%)
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.bar(model_labels, df_results["MAPE (%)"], color=["#1abc9c", "#16a085", "#27ae60"], edgecolor="black", width=0.5)
        ax3.set_title("Perbandingan MAPE (%) - Persentase Error Relatif", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Persentase Error (%)", fontsize=10)
        ax3.grid(True, linestyle=":", alpha=0.6)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Panel 4: Perbandingan Prediksi vs Nilai Aktual (Sampel 40 Pertama)
        ax4 = fig.add_subplot(gs[1, 1])
        n_sample = 40
        sample_idx = np.arange(n_sample)
        ax4.plot(sample_idx, y_test.iloc[:n_sample].values, label="Aktual (pH Asli)", color="black", linewidth=2.5, marker="o", markersize=4)
        ax4.plot(sample_idx, predictions["Linear Regression (Baseline)"][:n_sample], label="Linear Reg", color="#3498db", linestyle=":", alpha=0.8)
        ax4.plot(sample_idx, predictions["Gradient Boosting Regressor"][:n_sample], label="Gradient Boost", color="#9b59b6", linestyle="--", alpha=0.8)
        ax4.plot(sample_idx, predictions["Random Forest Regressor (Juara)"][:n_sample], label="Random Forest (Juara)", color="#2ecc71", linewidth=2)
        ax4.set_title(f"Visualisasi Prediksi vs Aktual (40 Sampel Pertama Data Uji)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Indeks Sampel Uji", fontsize=10)
        ax4.set_ylabel("Nilai pH", fontsize=10)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("HASIL KOMPARASI LENGKAP 3 MODEL REGRESI (KELOMPOK 3)", fontsize=15, fontweight="bold", y=0.98)
        plt.savefig(comp_plot_path, dpi=300)
        plt.close()
        print(f"[5] Grafik komparasi menyeluruh disimpan ke: {comp_plot_path}")

    print("\n" + "=" * 70)
    print(" [SELESAI] Komparasi Model Berhasil Diselesaikan! ")
    print("=" * 70 + "\n")

    return df_results, best_model_name

if __name__ == "__main__":
    run_regression_comparison(save_visualizations=True, export_best=True)
