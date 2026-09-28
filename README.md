# UTS Machine Learning - Water Potability

Klasifikasi kualitas air menggunakan Decision Tree dengan perbandingan metode resampling (SMOTE vs RUS).

## Dataset
Water Quality / Potability dari [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Pembagian Tugas
| Anggota | File | Bagian |
|---------|------|--------|
| Adel | `preprocessing.py` | Preprocessing + Split Data + Simple Feature Scaling |
| Juanne | `smote_pipeline.py` | SMOTE → Decision Tree → Training → Testing → Evaluasi |
| Risha | `rus_pipeline.py` | RUS → Decision Tree → Training → Testing → Evaluasi |

## Cara Menjalankan
```bash
pip install -r requirements.txt
python main.py
```