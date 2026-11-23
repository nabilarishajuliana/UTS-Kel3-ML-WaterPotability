# ============================================
# MAIN - JANGAN DIEDIT
# ============================================
# File ini menjalankan seluruh pipeline:
# Preprocessing → Bi-LSTM Training → Evaluasi Pipeline
# ============================================

import os
import sys
import warnings
import tempfile

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_HOME'] = os.path.join(tempfile.gettempdir(), '.keras')
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), '.matplotlib')

from preprocessing_lstm import load_and_preprocess
from bilstm_model_pipeline import run_bilstm_pipeline
from evaluation_pipeline import run_evaluation_pipeline

if __name__ == "__main__":
    # Step 1: Preprocessing & Time-Series Windowing (Adel)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess()

    # Step 2: Stacked Bi-LSTM Training (Juanne)
    model, history = run_bilstm_pipeline(X_train, y_train, X_val, y_val)

    # Step 3: Evaluasi, Denormalisasi & Visualisasi (Risha)
    run_evaluation_pipeline(model, X_test, y_test, scaler, history)

    print("\n" + "=" * 50)
    print("SELESAI - Seluruh pipeline Bi-LSTM berhasil dieksekusi")
    print("=" * 50)
