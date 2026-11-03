# Pertemuan 11 — Analisis Regresi Kualitas Air (Water Quality Regression)
### Kelompok 3 | Machine Learning
**Studi Kasus:** Prediksi Tingkat Keasaman Air (*pH Value*) Berdasarkan Kandungan Mineral & Senyawa Kimia

---

## 📌 1. Latar Belakang & Konsep Penggunaan Dataset

### Mengapa Dataset Water Potability Digunakan untuk Regresi?
1. **Syarat Regresi:** Variabel target ($y$) wajib berupa **data numerik kontinu** (*continuous real numbers*), bukan kategori/biner (0 atau 1).
2. **Transformasi Target Data:**
   - Pada tugas sebelumnya (UTS & Pertemuan 8), kolom biner `Potability` (0/1) dijadikan target klasifikasi.
   - Pada tugas TM 11 Regresi ini, kita memprediksi parameter fisik-kimia penting kualitas air, yaitu **`ph` (derajat keasaman air)** sebagai **Target ($y$)**.
   - Kolom fitur input ($X$) adalah 8 parameter zat kimia lainnya:
     `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, dan `Turbidity`.
   - Kolom `Potability` di-*drop* agar pemodelan murni mempelajari dinamika fisik-kimiawi air.
3. **Keuntungan untuk Kelompok 3:**
   - Menjaga kontinuitas tema proyek (*Water Potability*) dari awal perkuliahan.
   - Bersifat orisinal dan unik dibanding kelompok lain (tidak memakai dataset pasaran seperti harga rumah/mobil).
   - Memenuhi arahan dosen: setiap kelompok memiliki dataset yang berbeda.

---

## 📊 2. Hasil Benchmark Awal & Karakteristik Data

Berdasarkan pengujian nyata (*benchmark*) terhadap 2.785 data air valid:

| Model Regresi | MAE | RMSE | MAPE | $R^2$ Score | Status Performa |
|---|---|---|---|---|---|
| **Random Forest Regressor** | **1.2169** | **1.5265** | **19.36%** | **0.0316** | 🏆 **Model Terbaik (Terpilih)** |
| **Gradient Boosting Regressor** | 1.2153 | 1.5493 | 19.36% | 0.0024 | Model Pembanding (Boosting) |
| **Linear Regression (Baseline)** | 1.2182 | 1.5539 | 19.43% | -0.0034 | Model Baseline (Linear) |

> [!NOTE]
> **Analisis Ilmiah untuk Laporan:**
> Matriks korelasi antar zat kimia pada dataset ini berkisar antara **-0.10 s.d +0.08** (saling independen / non-linear).
> Oleh karena itu, model berbasis garis lurus (*Linear Regression*) gagal menghasilkan $R^2$ positif. Sebaliknya, model **Random Forest** terbukti paling tangguh mempartisi batas non-linear data air dengan **RMSE terendah (1.5265)** dan **MAE 1.21** (rata-rata selisih prediksi dengan nilai pH aktual hanya sekitar $\pm 1.2$).

---

## 👥 3. Pembagian Tugas Kelompok 3 (Rata & Adil)

Pembagian kerja dibagi seimbang antara koding modul, pemodelan, perancangan aplikasi UI/UX, dan penulisan laporan akhir:

| Anggota | Modul / File Koding | Rincian Tanggung Jawab Teknis | Kontribusi Laporan (PDF) |
|---|---|---|---|
| **Adel** | `preprocessing_regresi.py`<br>`linear_regression_pipeline.py` | 1. Pembersihan data & penanganan baris target `ph`.<br>2. Imputasi missing values (`Sulfate`, `Trihalomethanes`) dengan median.<br>3. Outlier handling (IQR) & Feature Scaling.<br>4. Data splitting (80% train / 20% test).<br>5. Implementasi Baseline: **Linear Regression** & evaluasi awal. | - **Bab I:** Landasan Teori (Konsep Regresi & Linear Regression).<br>- **Bab II:** Proses Preprocessing & Baseline Model (diagram alur, formula, & screenshot eksekusi). |
| **Juanne** | `random_forest_pipeline.py`<br>`gradient_boosting_pipeline.py`<br>`compare_regression.py` | 1. Implementasi **Random Forest Regressor** (Model Terbaik).<br>2. Implementasi **Gradient Boosting Regressor** (Boosting Pembanding).<br>3. Kalkulasi 5 metrik evaluasi lengkap (MAE, MSE, RMSE, MAPE, $R^2$).<br>4. Skrip komparasi model + visualisasi scatter plot prediksi vs aktual & *Feature Importance*. | - **Bab I:** Landasan Teori (Ensemble Learning: Random Forest & Gradient Boosting).<br>- **Bab II:** Hasil Evaluasi & Komparasi Model (tabel 5 metrik, grafik komparasi, & analisis performa). |
| **Risha** | `app.py`<br>`model_export.py`<br>(Integrasi UI/UX) | 1. Export model terbaik (**Random Forest Regressor**) ke file `.pkl` / `.joblib`.<br>2. Pembuatan aplikasi dashboard interaktif berbasis **Streamlit**.<br>3. Desain antarmuka: form slider 8 parameter air, tombol prediksi pH, kalkulator kualitas keasaman air, dan display metrik model. | - **Bab I:** Landasan Teori (Perancangan UI/UX untuk Model Machine Learning).<br>- **Bab II:** Perancangan UI/UX & Pengujian Sistem (mockup antarmuka, panduan penggunaan, & Kesimpulan). |

---

## 🎯 4. Berapa Model yang Diuji & Mengapa? (Konsep Komparasi)

Dalam tugas TM 11 ini, kita menguji **3 model regresi secara *head-to-head***:
1. **Model 1: Linear Regression** *(Baseline Model)*
2. **Model 2: Gradient Boosting Regressor** *(Ensemble Learning - Boosting)*
3. **Model 3: Random Forest Regressor** *(Ensemble Learning - Bagging)*

### Mengapa Kita Menguji 3 Model Ini?
- **Memenuhi Instruksi Tugas Poin 6:** Soal meminta: *"Buatlah UI/UX dari model yang memiliki performa yang **terbaik**"*. Kata *terbaik* berarti kita wajib menguji dan membandingkan beberapa model menggunakan data uji yang sama secara objektif.
- **Alur Pembuktian Ilmiah yang Kuat untuk Laporan:**
  - **Uji Coba 1 (Linear Regression):** Kita uji apakah hubungan zat kimia air bisa dimodelkan dengan garis lurus sederhana. Hasilnya terbukti kurang cocok ($R^2$ negatif) karena parameter air memiliki interaksi non-linear.
  - **Uji Coba 2 & 3 (Gradient Boosting vs Random Forest):** Kita beralih ke algoritma mutakhir *Ensemble Learning*. Kedua model ini diadu menggunakan 5 metrik evaluasi.
  - **Penentuan Juara:** Model **Random Forest** terbukti mengungguli model lain dengan **RMSE terendah (1.5265)** dan **$R^2$ terbaik (0.0316)**, sehingga model inilah yang dipilih dan diekspor ke aplikasi web **Streamlit**.

### 5 Metrik Evaluasi Wajib (Sesuai Modul TM 11):
1. **Mean Absolute Error (MAE):** Rata-rata selisih absolut antara nilai pH aktual dan prediksi.
2. **Mean Squared Error (MSE):** Rata-rata kuadrat kesalahan prediksi.
3. **Root Mean Squared Error (RMSE):** Akar kuadrat dari MSE (skala sama dengan satuan pH).
4. **Mean Absolute Percentage Error (MAPE):** Rata-rata persentase kesalahan relatif.
5. **R-Squared ($R^2$):** Mengukur proporsi variansi pH yang berhasil dijelaskan oleh model.

---

## 🌐 5. Perancangan UI/UX (Streamlit Web Dashboard)

Aplikasi Streamlit dibangun menggunakan model terbaik (**Random Forest**) dengan fitur:
- **Sidebar:** Informasi performa model terbaik (MAE, RMSE, MAPE, $R^2$) dan parameter input.
- **Form Input Interaktif:** Slider & input angka untuk 8 parameter kimia air.
- **Hasil Prediksi pH:**
  - Nilai prediksi angka pH (contoh: **7.15**).
  - Status kualitas air otomatis:
    - 🔴 **Asam** (pH < 6.5) — *Tidak aman dikonsumsi langsung*
    - 🟢 **Netral / Aman** (pH 6.5 - 8.5) — *Sesuai baku mutu air minum*
    - 🔵 **Basa** (pH > 8.5) — *Cenderung basa tinggi*
  - Visualisasi indikator posisi nilai pH pada grafik skala meter keasaman.

---

## 📁 6. Struktur Direktori Proyek

```text
Pertemuan11_Regresi/
│
├── data/
│   └── water_potability.csv             # Dataset kualitas air
├── preprocessing_regresi.py             # Data cleansing, median imputation, & split (Adel)
├── linear_regression_pipeline.py        # Baseline Linear Regression (Adel)
├── random_forest_pipeline.py            # Random Forest Regressor - Model Juara (Juanne)
├── gradient_boosting_pipeline.py        # Gradient Boosting Regressor (Juanne)
├── compare_regression.py                # Evaluasi 5 metrik & visualisasi komparasi (Juanne)
├── main_regresi.py                      # Runner seluruh pipeline regresi sekaligus
├── app.py                               # Dashboard interaktif Streamlit UI/UX (Risha)
├── requirements.txt                     # Dependensi Python
├── README.md                            # Dokumentasi modul pertemuan 11
└── Tugas TM11_Kelompok 3.pdf            # Laporan akhir praktikum
```

---

## ⏰ 7. Ketentuan Pengumpulan Tugas

- **Nama File Laporan:** `Tugas TM11_Kelompok 3.pdf`
- **Isi Pengumpulan:** File Laporan PDF + Seluruh Source Code (kodingan)
- **Batas Waktu Pengumpulan:** Minggu, 8 November 2026 pukul 23.59 WIB
