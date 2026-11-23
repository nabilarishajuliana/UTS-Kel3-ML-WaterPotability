import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main.py
# ---------------------------------------------------------------
def run_evaluation_pipeline(model, X_test, y_test, scaler, history):
    """
    Pipeline: Prediksi -> Denormalisasi -> Hitung Metrik -> Visualisasi
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(base_dir, "visualisasi")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 50)
    print("EVALUASI & VISUALISASI PIPELINE (RISHA)")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. PREDIKSI PADA DATA TESTING
    # ------------------------------------------------------------------
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    print(f"\n[1] Prediksi data testing selesai: {len(y_pred_scaled)} sampel")

    # ------------------------------------------------------------------
    # 2. DENORMALISASI KE SATUAN ASLI (UG/M3)
    # ------------------------------------------------------------------
    target_min = scaler.data_min_[0]
    target_max = scaler.data_max_[0]
    y_actual = y_test * (target_max - target_min) + target_min
    y_pred   = y_pred_scaled * (target_max - target_min) + target_min
    print(f"[2] Denormalisasi target selesai (kembali ke skala ug/m3)")

    # ------------------------------------------------------------------
    # 3. HITUNG METRIK EVALUASI REGRESI DERET WAKTU
    # ------------------------------------------------------------------
    mse  = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_actual, y_pred)
    r2   = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / np.maximum(y_actual, 1e-5))) * 100

    print("\n[3] HASIL EVALUASI MODEL PADA DATA UJI:")
    print(f"    Root Mean Squared Error (RMSE) : {rmse:.4f} ug/m3")
    print(f"    Mean Absolute Error (MAE)      : {mae:.4f} ug/m3")
    print(f"    R-Squared Score (R2)           : {r2:.4f} ({r2*100:.2f}%)")
    print(f"    Mean Absolute Percentage Error : {mape:.2f}%")

    # ------------------------------------------------------------------
    # 4. VISUALISASI: KURVA LOSS & MAE
    # ------------------------------------------------------------------
    hist_path = os.path.join(vis_dir, 'grafik_pelatihan_lstm.png')
    epochs_range = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(13, 5), dpi=300)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss (MSE)', color='#1f77b4', lw=2.2)
    plt.plot(epochs_range, history.history['val_loss'], label='Val Loss (MSE)', color='#ff7f0e', lw=2.2, linestyle='--')
    plt.title('Kurva Loss (MSE) - Stacked Bi-LSTM', fontsize=11, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Mean Squared Error', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['mae'], label='Train MAE', color='#2ca02c', lw=2.2)
    plt.plot(epochs_range, history.history['val_mae'], label='Val MAE', color='#d62728', lw=2.2, linestyle='--')
    plt.title('Kurva MAE - Stacked Bi-LSTM', fontsize=11, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Mean Absolute Error', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"\n[4] Grafik kurva pelatihan disimpan ke: {hist_path}")

    # ------------------------------------------------------------------
    # 5. VISUALISASI: GARIS AKTUAL VS PREDIKSI
    # ------------------------------------------------------------------
    plot_points = 200
    pred_path = os.path.join(vis_dir, 'grafik_prediksi_vs_aktual.png')
    plt.figure(figsize=(14, 5.5), dpi=300)
    hours = np.arange(1, plot_points + 1)
    plt.plot(hours, y_actual[:plot_points], label='Nilai Aktual Benzena (ug/m3)',
             color='#1f77b4', lw=2, marker='o', markersize=3, alpha=0.85)
    plt.plot(hours, y_pred[:plot_points], label='Prediksi Stacked Bi-LSTM (ug/m3)',
             color='#d62728', lw=2, linestyle='--', marker='x', markersize=3, alpha=0.9)
    plt.title(f'Perbandingan Nilai Aktual vs Prediksi Konsentrasi Benzena C6H6(GT)\n(Sampel {plot_points} Jam Pengujian - R2 Score: {r2:.4f} | Penanggung Jawab: Nabila Risha)',
              fontsize=11, fontweight='bold')
    plt.xlabel('Jam Pengujian ke-', fontsize=10)
    plt.ylabel('Konsentrasi Benzena (ug/m3)', fontsize=10)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(pred_path)
    plt.close()
    print(f"[5] Grafik prediksi vs aktual disimpan ke: {pred_path}")

    # ------------------------------------------------------------------
    # 6. VISUALISASI: SCATTER PLOT REGRESI
    # ------------------------------------------------------------------
    scatter_path = os.path.join(vis_dir, 'grafik_scatter_evaluasi.png')
    plt.figure(figsize=(6.5, 6), dpi=300)
    plt.scatter(y_actual, y_pred, color='#2b5c8f', alpha=0.45, edgecolors='none', s=25, label='Data Uji')
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Garis Ideal (y = x)')
    plt.title(f'Scatter Plot: Aktual vs Prediksi Bi-LSTM\nR2 Score: {r2:.4f} | RMSE: {rmse:.4f}',
              fontsize=11, fontweight='bold')
    plt.xlabel('Nilai Aktual Benzena (ug/m3)', fontsize=10)
    plt.ylabel('Nilai Prediksi Model (ug/m3)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    print(f"[6] Scatter plot evaluasi disimpan ke: {scatter_path}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
