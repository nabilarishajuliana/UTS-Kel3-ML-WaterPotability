# Pertemuan 8 — Klasifikasi Air (Water Potability)
### Kelompok 3 | Machine Learning

---

## 📋 Deskripsi Tugas

Mengimplementasikan dan membandingkan **3 algoritma klasifikasi** untuk memprediksi kelayakan air minum menggunakan dataset Water Potability:

| Model | Keterangan |
|---|---|
| **Naive Bayes** | Menggunakan GaussianNB (asumsi distribusi normal Gaussian) |
| **SVM (Support Vector Machine)** | Kernel RBF (Radial Basis Function) |
| **Decision Tree** | Baseline pembanding dari hasil tugas UTS |

---

## 👥 Pembagian Tugas Kelompok 3

| Anggota | File Utama | Tanggung Jawab & Bagian |
|---|---|---|
| **Adel** | `preprocessing_p8.py`, `app.py` | Preprocessing, Simple Feature Scaling ($X/X_{max}$), Split Berstrata 80:20, Implementasi UI/UX Web Dashboard Streamlit |
| **Juanne** | `naive_bayes_pipeline.py`, `compare_models.py` | Pipeline Gaussian Naive Bayes (SMOTE & RUS), Evaluasi Metrik & Visualisasi Komparasi |
| **Risha** | `svm_pipeline.py`, `decision_tree_pipeline.py` | Pipeline Support Vector Machine (Kernel RBF - SMOTE & RUS), Evaluasi Baseline UTS Decision Tree |

---


## 📁 Struktur Folder

```
Pertemuan8_Klasifikasi/
├── preprocessing_p8.py       # Load & preprocessing dataset
├── naive_bayes_pipeline.py   # Pipeline Naive Bayes
├── svm_pipeline.py           # Pipeline SVM (kernel RBF)
├── decision_tree_pipeline.py # Pipeline Decision Tree
├── compare_models.py         # Perbandingan ketiga model
├── main_p8.py                # Jalankan semua pipeline
├── app.py                    # Aplikasi Streamlit (UI)
└── requirements.txt          # Dependensi Python
```

---

## ▶️ Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan semua pipeline (CLI)
```bash
python main_p8.py
```

### 3. Jalankan aplikasi Streamlit (UI)
```bash
streamlit run app.py
```

---

## 📊 Preprocessing

Sama seperti UTS:
1. Load dataset `water_potability.csv`
2. Isi missing value dengan **median** tiap kolom
3. Handle outlier dengan **IQR clip**
4. Feature scaling: **X / max**
5. Split **80% train / 20% test** (stratified)

---

## 🎯 Metrik Evaluasi

Setiap model dievaluasi dengan:
- **Accuracy** — proporsi prediksi benar keseluruhan
- **Precision** — dari yang diprediksi layak, berapa yang benar layak
- **Recall** — dari yang benar layak, berapa yang berhasil terdeteksi
- **F1-Score** — rata-rata harmonis Precision & Recall

---

## 🌐 Aplikasi Streamlit

Fitur aplikasi:
- Form input 9 parameter kualitas air
- Pilih model (Naive Bayes / SVM / Decision Tree) di sidebar
- Tampilkan prediksi + probabilitas kelas
- Performa masing-masing model ditampilkan di sidebar
