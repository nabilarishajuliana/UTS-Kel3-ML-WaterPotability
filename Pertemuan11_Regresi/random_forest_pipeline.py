import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_regresi import load_and_preprocess_regression

def train_and_evaluate_random_forest(save_plot=True, export_model=True):
    """
    Melatih model Random Forest Regressor (Ensemble Bagging) sebagai model terbaik,
    menghitung Feature Importance, mengevaluasi 5 metrik, dan menyimpan visualisasi.

    Returns:
      model (RandomForestRegressor): Objek model yang telah dilatih.
      metrics (dict): Nilai 5 metrik evaluasi (MAE, MSE, RMSE, MAPE, R2).
      y_pred (np.ndarray): Hasil prediksi nilai pH pada data uji.
      feature_importance_df (pd.DataFrame): Kontribusi tiap fitur terhadap prediksi.
    """
    # 1. Ambil data yang telah dipreprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_regression(scale_features=True)

    # 2. Inisialisasi dan Latih Model Random Forest Regressor
    print("\n[1] Melatih Model Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("    -> Pelatihan model selesai (100 Decision Trees).")

    # 3. Prediksi pada Data Uji (Testing Set)
    print("\n[2] Melakukan Prediksi pada Data Uji (Testing Set)...")
    y_pred = model.predict(X_test)

    # 4. Hitung 5 Metrik Evaluasi Wajib (Modul TM 11)
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
    print(" HASIL EVALUASI 5 METRIK - RANDOM FOREST REGRESSOR ")
    print("=" * 50)
    print(f" 1. Mean Absolute Error (MAE)        : {mae:.4f}")
    print(f" 2. Mean Squared Error (MSE)         : {mse:.4f}")
    print(f" 3. Root Mean Squared Error (RMSE)   : {rmse:.4f}")
    print(f" 4. Mean Absolute Pct Error (MAPE)   : {mape:.2f}%")
    print(f" 5. R-Squared Score (R2)             : {r2:.4f}")
    print("=" * 50)

    # 5. Analisis Feature Importance (Kontribusi Fitur)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Fitur": X_train.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n[3] Tingkat Pengaruh Parameter Fitur (Feature Importance):")
    for _, row in feature_importance_df.iterrows():
        print(f"    - {row['Fitur']:18s}: {row['Importance']*100:6.2f}%")

    # 6. Ekspor Model ke File .joblib untuk Digunakan di UI/UX Streamlit
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if export_model:
        model_path = os.path.join(base_dir, "model_random_forest.joblib")
        joblib.dump(model, model_path)
        print(f"\n[5] Model terbaik berhasil diekspor ke: {model_path}")

    # 7. Visualisasi Hasil Evaluasi
    if save_plot:
        output_dir = os.path.join(base_dir, "visualisasi")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "evaluasi_random_forest.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Actual vs Predicted
        axes[0].scatter(y_test, y_pred, color="#16a085", alpha=0.6, edgecolors="k", s=35, label="Data Uji")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")
        axes[0].set_title("Actual vs Predicted pH\n(Random Forest Regressor)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Nilai pH Aktual", fontsize=10)
        axes[0].set_ylabel("Nilai pH Prediksi", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Subplot 2: Residual Plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, color="#2980b9", alpha=0.6, edgecolors="k", s=35)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
        axes[1].set_title("Residual Plot (Error vs Predicted)\nRandom Forest", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Nilai pH Prediksi", fontsize=10)
        axes[1].set_ylabel("Residual (Aktual - Prediksi)", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)

        # Subplot 3: Feature Importance
        sorted_fi = feature_importance_df.sort_values(by="Importance", ascending=True)
        axes[2].barh(sorted_fi["Fitur"], sorted_fi["Importance"], color="#27ae60", edgecolor="black")
        axes[2].set_title("Feature Importance (% Kontribusi)\nPengaruh terhadap pH", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Skor Kepentingan (Importance)", fontsize=10)
        axes[2].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[6] Visualisasi evaluasi disimpan di: {plot_path}")

    print("\n" + "=" * 65)
    print(" [SELESAI] Pipeline Random Forest Selesai Dijalankan! ")
    print("=" * 65 + "\n")

    return model, metrics, y_pred, feature_importance_df

if __name__ == "__main__":
    train_and_evaluate_random_forest(save_plot=True, export_model=True)

