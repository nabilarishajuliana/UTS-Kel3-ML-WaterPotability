# Tugas Praktikum Deep Learning: Peramalan Kualitas Udara Multivariat dengan Stacked Bi-LSTM

**Kelompok 3 - Kelas TI C2**
* Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga Surabaya
* Tahun Ajaran: 2025/2026

### Anggota Kelompok & Pembagian Tugas:
1. **Adella Putri Asy'ari** (NIM: 434241023)
   - Peran: *Data Preprocessing & Time-Series Windowing Engineer*
   - Berkas: `preprocessing_lstm.py` (Pemuatan data, interpolasi linier, normalisasi MinMaxScaler, analisis korelasi, dan sliding window 24 jam).
2. **Juanne Meilanie Raya Maturbong** (NIM: 434241026)
   - Peran: *Model Architecture & Bi-LSTM Training Engineer*
   - Berkas: `bilstm_model_pipeline.py` (Perancangan arsitektur Stacked Bi-LSTM, regularisasi Dropout 0.2, callbacks ModelCheckpoint & EarlyStopping, pelatihan model).
3. **Nabila Risha Juliana** (NIM: 434241017)
   - Peran: *Model Evaluation, Denormalization & Visualization Analyst*
   - Berkas: `evaluation_pipeline.py` (Pengujian inferensi, denormalisasi output ke µg/m³, evaluasi metrik RMSE/MAE/R², dan pembuatan seluruh grafik visualisasi).

---

## 1. Deskripsi Tugas
Proyek ini mengimplementasikan model Deep Learning berbasis **Stacked Bidirectional Long Short-Term Memory (Bi-LSTM)** untuk peramalan deret waktu multivariat (*multivariate time-series forecasting*). Model dilatih untuk memprediksi konsentrasi gas polutan karsinogenik **Benzena ($C_6H_6$)** satu jam ke depan ($t+1$) berdasarkan riwayat 24 jam ke belakang dari sensor gas dan variabel cuaca.

* **Sumber Dataset:** [Air Quality Data Set di Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set) / UCI Machine Learning Repository.
* **Jumlah Data:** 9.358 baris observasi per jam (Maret 2004 s.d. April 2005).
* **Partisi Data:** Time-series chronological split (70% Train, 15% Validation, 15% Test).

---

## 2. Struktur Direktori Proyek
```text
Tugas_LSTM_AirQuality/
├── data/
│   ├── AirQualityUCI.csv               # Dataset utama per jam
│   └── AirQualityUCI.xlsx              # Dokumen referensi Excel
├── model/
│   └── best_bilstm_airquality.keras    # Model bobot terbaik Bi-LSTM tersimpan
├── visualisasi/
│   ├── grafik_korelasi_fitur.png       # Heatmap korelasi antar variabel (Adel)
│   ├── grafik_pelatihan_lstm.png       # Kurva loss (MSE) & MAE pelatihan (Risha)
│   ├── grafik_prediksi_vs_aktual.png   # Grafik garis perbandingan aktual vs prediksi (Risha)
│   └── grafik_scatter_evaluasi.png     # Scatter plot regresi & garis ideal y = x (Risha)
├── preprocessing_lstm.py               # Modul 1 (Adel)   : Preprocessing & windowing
├── bilstm_model_pipeline.py            # Modul 2 (Juanne) : Arsitektur & pelatihan model
├── evaluation_pipeline.py              # Modul 3 (Risha)  : Evaluasi metrik & visualisasi
├── main.py                             # Runner utama pipeline terpadu
└── README.md                           # Dokumentasi proyek
```

---

## 3. Cara Menjalankan Program di Terminal
1. Buka terminal PowerShell / Command Prompt di direktori proyek ini:
   ```powershell
   cd d:\UTS-Kel3-ML-WaterPotability\Tugas_LSTM_AirQuality
   ```
2. Jalankan skrip runner utama:
   ```powershell
   python main.py
   ```
3. Program akan secara otomatis memproses dataset, melatih model Bi-LSTM selama 20 epoch (~18 detik), mengevaluasi metrik pada data uji, dan menyimpan model serta seluruh visualisasi grafik ke folder `visualisasi/`.

---

## 4. Ringkasan Hasil Evaluasi pada Data Uji (Testing Set)
* **Root Mean Squared Error (RMSE):** `2.8061 µg/m³`
* **Mean Absolute Error (MAE):** `1.9555 µg/m³`
* **R-Squared Score ($R^2$):** `0.7848` (**78.48%**)
* **Mean Absolute Percentage Error (MAPE):** `40.08%`
