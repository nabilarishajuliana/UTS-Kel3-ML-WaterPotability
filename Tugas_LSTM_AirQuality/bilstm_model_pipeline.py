import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ---------------------------------------------------------------
# FUNGSI UTAMA — dipanggil oleh main.py
# ---------------------------------------------------------------
def run_bilstm_pipeline(X_train, y_train, X_val, y_val):
    """
    Pipeline: Build Bi-LSTM -> Callbacks -> Training
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'best_bilstm_airquality.keras')

    print("=" * 50)
    print("STACKED BI-LSTM MODEL & TRAINING (JUANNE)")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. MEMBANGUN ARSITEKTUR STACKED BI-LSTM
    # ------------------------------------------------------------------
    print("\n[1] Membangun arsitektur Stacked Bi-LSTM...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        # Lapisan 1: Bidirectional LSTM memproses maju dan mundur
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        # Lapisan 2: LSTM standar untuk ekstraksi representasi sekuensial
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        # Regressor Head
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ], name="Stacked_BiLSTM_Kelompok3")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    # ------------------------------------------------------------------
    # 2. KONFIGURASI CALLBACKS (MODELCHECKPOINT & EARLYSTOPPING)
    # ------------------------------------------------------------------
    print("\n[2] Menyiapkan callbacks ModelCheckpoint & EarlyStopping...")
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)
    ]

    # ------------------------------------------------------------------
    # 3. PELATIHAN MODEL BI-LSTM
    # ------------------------------------------------------------------
    print("\n[3] Memulai pelatihan model Bi-LSTM (20 Epochs, Batch Size 64)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n    Model Bi-LSTM terbaik berhasil disimpan ke: {save_path}")
    return model, history
