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
├── Pertemuan8_Klasifikasi/               # Modul Tugas Pertemuan 8
│   ├── preprocessing_p8.py               # Preprocessing & Split Data
│   ├── naive_bayes_pipeline.py           # Pipeline Naive Bayes (GaussianNB)
│   ├── svm_pipeline.py                   # Pipeline SVM (RBF Kernel)
│   ├── decision_tree_pipeline.py         # Pipeline Decision Tree
│   ├── compare_models.py                 # Perbandingan metrik & visualisasi
│   ├── main_p8.py                        # Runner pipeline Pertemuan 8
│   ├── app.py                            # Aplikasi Web Prediksi (Streamlit)
│   └── README.md
└── Pertemuan11_Regresi/                  # Modul Tugas Pertemuan 11 (Regresi)
    ├── preprocessing_regresi.py          # Preprocessing + Splitting Data
    ├── linear_regression_pipeline.py     # Pipeline Baseline Linear Regression
    ├── random_forest_pipeline.py         # Pipeline Random Forest Regressor (Model Terbaik)
    ├── gradient_boosting_pipeline.py     # Pipeline Gradient Boosting Regressor
    ├── compare_regression.py             # Evaluasi 5 Metrik & Komparasi Model
    ├── main_regresi.py                   # Runner seluruh pipeline Pertemuan 11
    ├── app.py                            # Dashboard Prediksi pH (Streamlit UI/UX)
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

---

## 3. Tugas Pertemuan 11 (Regresi - Prediksi Nilai pH Air)
Memprediksi nilai kontinu tingkat keasaman air (**`ph`**) menggunakan **Linear Regression**, **Gradient Boosting Regressor**, dan **Random Forest Regressor** (model terbaik), dievaluasi dengan 5 metrik (MAE, MSE, RMSE, MAPE, $R^2$), serta diimplementasikan ke dalam UI/UX berbasis **Streamlit**.

### Pembagian Tugas Pertemuan 11:
| Anggota | File Utama | Rincian Tanggung Jawab |
|---|---|---|
| **Adel** | `preprocessing_regresi.py`, `linear_regression_pipeline.py` | Preprocessing, Imputasi Median, Split Data, Pipeline Baseline Linear Regression & Evaluasi Awal |
| **Juanne** | `random_forest_pipeline.py`, `gradient_boosting_pipeline.py`, `compare_regression.py` | Pipeline Random Forest (Model Juara), Gradient Boosting, Evaluasi 5 Metrik, & Visualisasi Komparasi |
### Cara Menjalankan Pertemuan 11:
```bash
cd Pertemuan11_Regresi
pip install -r requirements.txt

# Menjalankan seluruh pipeline regresi sekaligus:
python main_regresi.py

# Atau menjalankan modul per individu:
python preprocessing_regresi.py
python linear_regression_pipeline.py
python random_forest_pipeline.py
python gradient_boosting_pipeline.py
python compare_regression.py

# Menjalankan aplikasi web Streamlit:
streamlit run app.py
```
