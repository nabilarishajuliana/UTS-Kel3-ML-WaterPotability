# Catatan Briefing & Panduan Kelompok 3 — Praktikum Pertemuan 8 (Klasifikasi ML)

**Halo Nabila dan Juanne!** 👋  
Catatan ini aku buat sebagai panduan ringkas dan dokumentasi buat kelompok kita. Semua kode program, pengujian model, visualisasi, aplikasi web Streamlit, sampai laporan resmi (`.docx`) untuk tugas Pertemuan 8 **sudah aku selesaikan dan integrasikan seluruhnya**.

Setelah ini berkas ini bakal aku **push ke GitHub**, jadi kalian bisa langsung lakukan `git pull` di laptop masing-masing buat cek, jalanin, atau kalau ada yang mau kalian tambahin/sesuaikan sebelum dikumpulkan.

---

## 1. Pembagian Tugas Resmi Kita di Laporan & Repositori

Biar kita semua satu suara dan paham pas ditanya dosen atau pas evaluasi tugas, di Bab 2 Laporan (`LAPORAN_TUGAS_KLASIFIKASI.docx`) dan `README.md` repositori sudah tertulis pembagian tugas kita masing-masing:

| Anggota Kelompok | File / Modul | Bagian & Kontribusi yang Dikerjakan |
| :--- | :---: | :--- |
| **Adella Putri Asy'ari**<br>*(434241023)* | `preprocessing_p8.py`<br>`app.py` | Mengembangkan modul pembersihan data (imputasi median, penanganan outlier IQR clipping, simple feature scaling $X/X_{max}$, stratified split 80:20), perancangan antarmuka web interaktif Streamlit bertema *Fresh Aquatic*, penataan 3 kartu parameter fisik-kimia, dan fitur preset pengujian instan WHO. |
| **Juanne Meilanie Raya Maturbongs**<br>*(434241026)* | `naive_bayes_pipeline.py` | Mengembangkan pipeline klasifikasi Gaussian Naïve Bayes, melatih model pada data latih SMOTE (oversampling) dan RUS (undersampling), kalkulasi metrik evaluasi (Akurasi, Presisi, Recall, Spesifisitas, F1-Score), serta pembuatan visualisasi grafik Confusion Matrix dan diagram batang metrik. |
| **Nabila Risha Juliana**<br>*(434241017)* | `svm_pipeline.py`<br>`decision_tree_pipeline.py`<br>`compare_models.py`<br>`main_p8.py` | Mengembangkan pipeline klasifikasi Support Vector Machine (Kernel RBF), merekonstruksi model baseline Decision Tree UTS sebagai pembanding, membuat skrip orchestrator utama `main_p8.py`, melakukan komparasi 6 model di `compare_models.py`, serta ekspor tabel hasil evaluasi ke CSV. |

---

## 2. Cara Menjalankan Program di Laptop Kalian

Setelah kalian `git pull`, buka folder proyek kita di VS Code / terminal, lalu ikuti langkah mudah berikut:

### Langkah 1: Masuk ke folder Pertemuan 8
Pastikan terminal aktif di direktori `Pertemuan8_Klasifikasi`:
```bash
cd Pertemuan8_Klasifikasi
```

### Langkah 2: Jalankan Pipeline Evaluasi Model (Backend & Grafik)
Jalankan skrip orchestrator utama:
```bash
python main_p8.py
```
**Apa yang akan terjadi:**
- Terminal bakal otomatis ngejalanin pembersihan data, latih Naïve Bayes, latih SVM RBF, uji Decision Tree UTS, dan nampilin tabel rangkuman komparasi 6 model secara langsung di terminal.
- Otomatis nge-generate file grafik:
  - `viz_nb_1_confusion_matrix.png` & `viz_nb_2_metrik.png`
  - `viz_svm_1_confusion_matrix.png` & `viz_svm_2_metrik.png`
  - `viz_dt_1_confusion_matrix.png`
  - `viz_compare_all_models.png`
- Otomatis nyimpen tabel hasil metrik ke `hasil_perbandingan_semua_model.csv`.

### Langkah 3: Menjalankan Aplikasi Web Interaktif (UI Dashboard)
Untuk melihat tampilan visual dashboard web Streamlit:
```bash
streamlit run app.py
```
**Apa yang akan terjadi:**
- Browser otomatis kebuka ke alamat `http://localhost:8501`.
- Kalian bisa coba langsung input 9 parameter air, klik tombol preset instan (seperti *"Air Standar WHO"* atau *"Air Tercemar"*), lalu klik **"Prediksi Kelayakan Air"**.
- Sistem bakal ngeluarin status kelayakan (Layak Minum / Berbahaya) lengkap dengan persentase skor keyakinan model SVM RBF secara *real-time*.

---

## 3. Penjelasan Setiap File & Kenapa Hasilnya Begitu

Biar kita paham logika teknisnya dan siap kalau ditanya dosen, berikut penjelasan konsep per filenya:

### 1. `preprocessing_p8.py`
- **Kenapa file ini ada:** Supaya proses penyiapan data dilakukan terpusat di satu pintu. Semua model (Naïve Bayes, SVM, dan Decision Tree) wajib menggunakan data latih dan uji yang identik agar perbandingannya adil (*fair comparison*) dan tidak terjadi kebocoran data (*data leakage*).
- **Apa yang dikerjakan:** 
  1. Imputasi nilai kosong (*missing values*) pada kolom pH, Sulfate, dan Trihalomethanes menggunakan nilai **median** masing-masing kolom (karena distribusi data condong miring/skewed).
  2. Batasi outlier dengan metode **IQR Clipping** (dipotong ke batas bawah/atas, bukan dihapus barisnya agar jumlah sampel data tidak hilang).
  3. **Simple Feature Scaling** ($X / X_{max}$) agar nilai semua parameter berada pada rentang $[0, 1]$. Ini krusial banget buat SVM yang sensitif terhadap skala data.
  4. **Stratified Train-Test Split (80:20)** dengan `random_state=42` agar proporsi kelas target seimbang antara 2.620 data latih dan 656 data uji.

### 2. `naive_bayes_pipeline.py`
- **Kenapa file ini ada:** Ini adalah algoritma terawasi pertama yang diwajibkan di modul praktikum Pertemuan 8. Kita pakai varian **Gaussian Naïve Bayes** karena fitur parameter fisik-kimia air berupa data numerik kontinu.
- **Kenapa hasilnya seperti itu:**
  - Akurasi Naïve Bayes: **58,23%** (pada SMOTE) dan **57,62%** (pada RUS).
  - *Kenapa bagus:* Performanya meningkat signifikan dibandingkan baseline Decision Tree UTS (+1,8% s/d +2,6%), dan waktu komputasinya super instan (hanya beberapa milidetik).
  - *Kelemahannya:* Naïve Bayes mengasumsikan setiap zat kimia dalam air bersifat independen satu sama lain (asumsi *conditional independence*). Padahal di alam nyata, padatan terlarut (TDS/Solids) berhubungan erat dengan daya hantar listrik (Conductivity). Karena asumsi ini tidak terpenuhi sempurna di data riil, akurasinya masih di bawah SVM.

### 3. `svm_pipeline.py`
- **Kenapa file ini ada:** Ini adalah algoritma terawasi kedua di Pertemuan 8. Kita menggunakan **Support Vector Machine dengan Kernel Radial Basis Function (RBF)**.
- **Kenapa hasilnya seperti itu (JUARA 1):**
  - **SVM (RBF) + RUS**: Meraih **Akurasi Puncak 61,43%** dan **Spesifisitas 71,00%** (Tertinggi di antara semua model).
  - **SVM (RBF) + SMOTE**: Meraih **F1-Score Puncak 52,73%** dan **Recall 56,64%**.
  - *Alasan ilmiahnya:* Hubungan antar zat kimia pada kualitas air itu sifatnya kompleks dan non-linear (tidak bisa dipisahkan garis lurus). Kernel RBF secara matematis memetakan fitur ke ruang dimensi yang lebih tinggi sehingga berhasil membentuk bidang pemisah (*hyperplane*) melengkung yang sangat efektif memisahkan air minum dan air tercemar tanpa gampang terpengaruh noise data.

### 4. `decision_tree_pipeline.py`
- **Kenapa file ini ada:** Sebagai pembanding dasar (*baseline model*) yang mereplikasi eksperimen kita saat UTS kemarin. Dengan menguji Decision Tree pada 656 data uji yang sama persis, kita bisa membuktikan peningkatan kinerja model baru di Pertemuan 8 secara ilmiah.
- **Kenapa hasilnya seperti itu:**
  - Akurasi Decision Tree: **56,40%** (SMOTE) dan **55,03%** (RUS).
  - Merupakan model dengan performa terendah karena pohon keputusan tunggal tanpa pemangkasan (*unpruned tree*) sangat sensitif terhadap variasi data latih, membuat pemotongan ruang fiturnya kaku (sejajar sumbu), dan mudah mengalami *overfitting*.

### 5. `compare_models.py` & `main_p8.py`
- **Kenapa file ini ada:** `compare_models.py` bertugas mengumpulkan seluruh metrik evaluasi dari ke-6 variasi model, membuat grafik batang komparasi, dan menyimpannya ke tabel CSV. Sedangkan `main_p8.py` adalah skrip pengontrol (*orchestrator*) agar siapa pun yang mau menguji proyek kita cukup mengetik 1 perintah saja tanpa perlu repot menjalankan skrip satu demi satu.

### 6. `app.py`
- **Kenapa file ini ada:** Untuk memenuhi butir 6 panduan penugasan (implementasi antarmuka pengujian interaktif). Aplikasi ini mengintegrasikan model terbaik kita, yaitu SVM Kernel RBF.
- **Fitur Desain UI:** Menggunakan tema *Fresh Aquatic* (nuansa biru air laboratorium yang bersih, rapi, dan profesional tanpa emoji santai), input 9 fitur terkelompok dalam 3 kartu modular, dan dilengkapi tombol preset instan standar WHO.

### 7. `LAPORAN_TUGAS_KLASIFIKASI.docx`
- Dokumen laporan resmi kelompok yang sudah disesuaikan dengan format standar akademik laporan UTS Kelompok 3:
  - Font: **Times New Roman 100%**.
  - Warna font: **Hitam pekat (`RGB: 0, 0, 0`)** di seluruh isi, judul, dan tabel.
  - Perataan: **Rata Kanan dan Kiri (Justified)**.
  - **Format Tabel**: Sesuai format standar laporan dengan garis kisi penuh (*full black grid border*), kepala tabel abu-abu, dan sel data putih bersih.
  - Terdapat sub-bab **1.5 Manajemen Folder & Struktur Direktori** lengkap dengan diagram hierarki dan fungsi tiap berkas.

### 8. `Tugas klasifikasi_434241023.zip`
- Arsip pengumpulan akhir yang sudah dikemas rapi sesuai format penamaan resmi di modul praktikum untuk diserahkan ke dosen.

---

## 4. Tabel Hasil Evaluasi Lengkap Buat Bahan Diskusi Kita

Ini tabel ringkasan benchmark performa ke-6 model yang bisa kalian jadikan referensi saat diskusi atau presentasi:

| Rank | Model Klasifikasi | Resampling | Akurasi | Presisi | Recall | Spesifisitas | F1-Score | Karakteristik Utama |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Support Vector Machine (RBF)** | **RUS** | **61,43%** | **50,64%** | 46,48% | **71,00%** | 48,47% | **Akurasi & Spesifisitas Puncak** (Paling aman menolak air beracun) |
| 🥈 | **Support Vector Machine (RBF)** | **SMOTE** | **60,37%** | 49,32% | **56,64%** | 62,75% | **52,73%** | **F1-Score & Recall Tertinggi** (Paling andal mendeteksi air layak) |
| 🥉 | **Gaussian Naïve Bayes** | **SMOTE** | **58,23%** | 46,92% | 53,52% | 61,25% | 50,00% | Komputasi sangat cepat, mengungguli baseline UTS |
| 4 | **Gaussian Naïve Bayes** | **RUS** | **57,62%** | 45,83% | 47,27% | 64,25% | 46,54% | Kemampuan menolak air tercemar cukup baik |
| 5 | **Decision Tree (UTS)** | **SMOTE** | **56,40%** | 44,83% | 50,78% | 60,00% | 47,62% | Model baseline pembanding dari proyek UTS |
| 6 | **Decision Tree (UTS)** | **RUS** | **55,03%** | 43,65% | 52,34% | 56,75% | 47,60% | False Positive tertinggi, rentan overfitting |

---

## 5. Next Action Buat Kalian

1. Silakan lakukan `git pull` di laptop masing-masing.
2. Coba jalankan `python main_p8.py` dan `streamlit run app.py` untuk memastikan di lingkungan laptop kalian berjalan lancar tanpa kendala dependensi.
3. Buka dan cek dokumen [LAPORAN_TUGAS_KLASIFIKASI.docx](LAPORAN_TUGAS_KLASIFIKASI.docx). Jika ada kalimat atau bagian analisis yang menurut kalian mau ditambahkan/diperhalus, kabari aku ya agar bisa kita sinkronkan.

Semangat teman-teman! Proyek Pertemuan 8 kita sudah solid dan siap dikumpulkan. 🚀
