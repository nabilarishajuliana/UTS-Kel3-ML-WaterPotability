# Proyek Machine Learning - Water Potability (Kelompok 3)

Repositori ini berisi tugas Machine Learning untuk klasifikasi kelayakan air (*Water Potability*), yang terbagi ke dalam dua modul tugas:

## 📁 Struktur Direktori
```text
├── data/                                 # Dataset water_potability.csv
├── UTS_Kel3_ML_WaterPotability/          # Modul Tugas UTS
│   ├── preprocessing.py                  # Preprocessing + Scaling
│   ├── smote_pipeline.py                 # SMOTE + Decision Tree
│   ├── rus_pipeline.py                   # RUS + Decision Tree
│   ├── main.py                           # Runner seluruh pipeline UTS
│   └── requirements.txt
└── Pertemuan8_Klasifikasi/               # Modul Tugas Pertemuan 8
    ├── preprocessing_p8.py               # Preprocessing & Split Data
    ├── naive_bayes_pipeline.py           # Pipeline Naive Bayes (GaussianNB)
    ├── svm_pipeline.py                   # Pipeline SVM (RBF Kernel)
    ├── decision_tree_pipeline.py         # Pipeline Decision Tree
    ├── compare_models.py                 # Perbandingan metrik & visualisasi
    ├── main_p8.py                        # Runner pipeline Pertemuan 8
    ├── app.py                            # Aplikasi Web Prediksi (Streamlit)
    └── README.md
```

---

## 1. Tugas UTS (SMOTE vs RUS - Decision Tree)
Klasifikasi kualitas air menggunakan Decision Tree dengan perbandingan metode penanganan data tidak seimbang (SMOTE vs RUS).

### Pembagian Tugas UTS:
| Anggota | File | Bagian |
|---|---|---|
| Adel | `preprocessing.py` | Preprocessing + Split Data + Simple Feature Scaling |
| Juanne | `smote_pipeline.py` | SMOTE → Decision Tree → Training → Testing → Evaluasi |
| Risha | `rus_pipeline.py` | RUS → Decision Tree → Training → Testing → Evaluasi |

### Cara Menjalankan UTS:
```bash
cd UTS_Kel3_ML_WaterPotability
pip install -r requirements.txt
python main.py
```

---

## 2. Tugas Pertemuan 8 (Klasifikasi & Streamlit App)
Membandingkan 3 algoritma klasifikasi (**Naive Bayes**, **SVM**, dan **Decision Tree**) serta implementasi model terbaik dalam bentuk aplikasi web interaktif berbasis **Streamlit**.

### Cara Menjalankan Pertemuan 8:
```bash
cd Pertemuan8_Klasifikasi
pip install -r requirements.txt

# Menjalankan perbandingan model:
python main_p8.py

# Menjalankan aplikasi web Streamlit:
streamlit run app.py
```