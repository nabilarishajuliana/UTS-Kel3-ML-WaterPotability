import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# TUGAS PERTEMUAN 11 - ANALISIS REGRESI KUALITAS AIR
# Kelompok 3 (Machine Learning)
# Bagian: Dashboard Prediksi pH Interaktif (Streamlit UI/UX)
# Penanggung Jawab: Risha (Didukung oleh Adel & Juanne)
# ==============================================================================

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Nilai pH Air — Kelompok 3",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Desain CSS Tema Modern: Aqua Marine / Water Quality
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(180deg, #f0fdfa 0%, #e0f2fe 35%, #f8fafc 100%) !important;
        color: #0f172a !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1250px !important;
    }

    /* Header Hero Card */
    .header-card {
        background: #ffffff;
        border: 1px solid #bae6fd;
        border-radius: 16px;
        padding: 1.6rem 2rem;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.08), 0 4px 6px -2px rgba(2, 132, 199, 0.03);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .header-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #10b981 100%);
    }

    .header-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0369a1;
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .header-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0c4a6e;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.025em;
    }

    .header-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }

    /* Group Cards for Inputs */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 15px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
        height: 100%;
    }

    .input-card-header {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #0284c7;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Result Card Styles */
    .result-box-acid {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(244, 63, 94, 0.15);
    }

    .result-box-neutral {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.15);
    }

    .result-box-alkali {
        background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15);
    }

    .ph-display-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    .sidebar-metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. Fungsi Load Models & Scaler
@st.cache_resource
def load_all_artifacts():
    models = {}
    
    # 1. Random Forest (Model Juara)
    rf_path = os.path.join(BASE_DIR, "model_terbaik.joblib")
    if not os.path.exists(rf_path):
        rf_path = os.path.join(BASE_DIR, "model_random_forest.joblib")
    if os.path.exists(rf_path):
        models["Random Forest Regressor (Model Juara 🏆)"] = joblib.load(rf_path)

    # 2. Gradient Boosting
    gb_path = os.path.join(BASE_DIR, "model_gradient_boosting.joblib")
    if os.path.exists(gb_path):
        models["Gradient Boosting Regressor"] = joblib.load(gb_path)

    # 3. Linear Regression (Baseline)
    lr_path = os.path.join(BASE_DIR, "model_linear_regression.joblib")
    if os.path.exists(lr_path):
        models["Linear Regression (Baseline)"] = joblib.load(lr_path)

    # Scaler
    scaler_path = os.path.join(BASE_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # CSV Komparasi Metrik
    csv_path = os.path.join(BASE_DIR, "hasil_komparasi_regresi.csv")
    metrics_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    return models, scaler, metrics_df

models_dict, scaler, metrics_df = load_all_artifacts()

# ==============================================================================
# SIDEBAR: PENGATURAN & INFORMASI TIM
# ==============================================================================
with st.sidebar:
    st.markdown("### 💧 Laboratorium Kualitas Air")
    st.caption("Universitas / Praktikum Machine Learning")
    st.markdown("---")

    st.markdown("#### ⚙️ Pilihan Model Regresi")
    if models_dict:
        selected_model_name = st.selectbox(
            "Pilih Algoritma Regresi:",
            list(models_dict.keys()),
            index=0,
            help="Pilih model machine learning yang digunakan untuk memprediksi nilai pH."
        )
        active_model = models_dict[selected_model_name]
    else:
        st.error("Model belum dilatih! Jalankan `main_regresi.py` terlebih dahulu.")
        st.stop()

    # Tampilkan Metrik Model Terpilih
    if metrics_df is not None:
        st.markdown("#### 📊 Performa Model Terpilih (Testing Set)")
        
        match_row = None
        for _, row in metrics_df.iterrows():
            if "Random Forest" in selected_model_name and "Random Forest" in row["Model"]:
                match_row = row
                break
            elif "Gradient Boosting" in selected_model_name and "Gradient Boosting" in row["Model"]:
                match_row = row
                break
            elif "Linear Regression" in selected_model_name and "Linear Regression" in row["Model"]:
                match_row = row
                break

        if match_row is not None:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="sidebar-metric-card">
                    <small style="color:#64748b; font-weight:600;">MAE</small>
                    <div style="font-family:'JetBrains Mono'; font-size:1.15rem; font-weight:700; color:#0284c7;">
                        {match_row['MAE']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-metric-card">
                    <small style="color:#64748b; font-weight:600;">MAPE</small>
                    <div style="font-family:'JetBrains Mono'; font-size:1.15rem; font-weight:700; color:#0284c7;">
                        {match_row['MAPE (%)']:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="sidebar-metric-card">
                    <small style="color:#64748b; font-weight:600;">RMSE</small>
                    <div style="font-family:'JetBrains Mono'; font-size:1.15rem; font-weight:700; color:#0284c7;">
                        {match_row['RMSE']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-metric-card">
                    <small style="color:#64748b; font-weight:600;">R² Score</small>
                    <div style="font-family:'JetBrains Mono'; font-size:1.15rem; font-weight:700; color:#0284c7;">
                        {match_row['R2 Score']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("📌 Dievaluasi terhadap 557 sampel uji valid.")

    st.markdown("---")
    st.markdown("#### 👥 Tim Penyusun (Kelompok 3)")
    st.markdown("""
    - **Adel:** Preprocessing Data & Baseline Regresi
    - **Juanne:** Ensemble Model (RF & GB) + Komparasi
    - **Risha:** Perancangan UI/UX Dashboard Streamlit
    """)

# ==============================================================================
# HEADER HERO CARD
# ==============================================================================
st.markdown("""
<div class="header-card">
    <div class="header-badge">TUGAS MANDIRI 11 • ANALISIS REGRESI ENSEMBLE</div>
    <h1 class="header-title">🌊 Sistem Prediksi Derajat Keasaman (pH) Air</h1>
    <p class="header-subtitle">
        Aplikasi estimasi nilai <b>pH air secara kontinu</b> berdasarkan 8 parameter fisik-kimiawi air
        menggunakan perbandingan algoritma <b>Linear Regression</b>, <b>Gradient Boosting</b>, dan <b>Random Forest Regressor</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TAB NAVIGASI UTAMA
# ==============================================================================
tab_prediksi, tab_komparasi, tab_edukasi = st.tabs([
    "🚀 Kalkulator Prediksi Interaktif",
    "📊 Hasil Komparasi Model & Metrik",
    "📘 Panduan Baku Mutu Air (Permenkes / WHO)"
])

# ------------------------------------------------------------------------------
# TAB 1: KALKULATOR PREDIKSI INTERAKTIF
# ------------------------------------------------------------------------------
with tab_prediksi:
    st.markdown("#### ⚡ Preset Sampel Air Cepat:")
    st.caption("Klik salah satu tombol preset di bawah untuk mengisi formulir secara instan:")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    default_values = {
        "Hardness": 196.37,
        "Solids": 20927.83,
        "Chloramines": 7.13,
        "Sulfate": 333.78,
        "Conductivity": 421.88,
        "Organic_carbon": 14.28,
        "Trihalomethanes": 66.40,
        "Turbidity": 3.96
    }

    if col_p1.button("🟢 Air Baku Mutu Normal (Netral)", use_container_width=True):
        st.session_state["Hardness"] = 180.0
        st.session_state["Solids"] = 15000.0
        st.session_state["Chloramines"] = 6.2
        st.session_state["Sulfate"] = 300.0
        st.session_state["Conductivity"] = 380.0
        st.session_state["Organic_carbon"] = 11.5
        st.session_state["Trihalomethanes"] = 50.0
        st.session_state["Turbidity"] = 3.2

    if col_p2.button("🔴 Air Indikasi Asam (Tinggi Sulfat)", use_container_width=True):
        st.session_state["Hardness"] = 120.0
        st.session_state["Solids"] = 29000.0
        st.session_state["Chloramines"] = 8.9
        st.session_state["Sulfate"] = 430.0
        st.session_state["Conductivity"] = 550.0
        st.session_state["Organic_carbon"] = 19.5
        st.session_state["Trihalomethanes"] = 80.0
        st.session_state["Turbidity"] = 5.1

    if col_p3.button("🔵 Air Sadah Tinggi (Indikasi Basa)", use_container_width=True):
        st.session_state["Hardness"] = 280.0
        st.session_state["Solids"] = 17500.0
        st.session_state["Chloramines"] = 4.8
        st.session_state["Sulfate"] = 260.0
        st.session_state["Conductivity"] = 340.0
        st.session_state["Organic_carbon"] = 9.8
        st.session_state["Trihalomethanes"] = 40.0
        st.session_state["Turbidity"] = 2.8

    if col_p4.button("🔄 Reset Rata-rata Dataset", use_container_width=True):
        for k, v in default_values.items():
            st.session_state[k] = v

    st.markdown("<br>", unsafe_allow_html=True)

    # Form Parameter Input Terbagi dalam 2 Card Rapih
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        <div class="input-card-header">
            <span>🧪</span> Parameter Mineral & Zat Kimia Terlarut
        </div>
        """, unsafe_allow_html=True)
        
        hardness = st.number_input(
            "1. Kesadahan / Hardness (mg/L)",
            min_value=40.0, max_value=350.0,
            value=float(st.session_state.get("Hardness", default_values["Hardness"])),
            step=1.0, help="Kandungan ion kalsium (Ca) dan magnesium (Mg)."
        )
        solids = st.number_input(
            "2. Total Padatan Terlarut / Solids (ppm)",
            min_value=300.0, max_value=65000.0,
            value=float(st.session_state.get("Solids", default_values["Solids"])),
            step=100.0, help="Total Dissolved Solids (TDS) mencakup garam anorganik."
        )
        chloramines = st.number_input(
            "3. Kloramin / Chloramines (ppm)",
            min_value=0.3, max_value=15.0,
            value=float(st.session_state.get("Chloramines", default_values["Chloramines"])),
            step=0.1, help="Desinfektan air turunan klorin dan amonia."
        )
        sulfate = st.number_input(
            "4. Sulfat / Sulfate (mg/L)",
            min_value=120.0, max_value=500.0,
            value=float(st.session_state.get("Sulfate", default_values["Sulfate"])),
            step=1.0, help="Kadar senyawa sulfat (SO4) terlarut dalam air."
        )

    with col_c2:
        st.markdown("""
        <div class="input-card-header">
            <span>⚡</span> Karakteristik Fisik & Senyawa Organik
        </div>
        """, unsafe_allow_html=True)

        conductivity = st.number_input(
            "5. Daya Hantar Listrik / Conductivity (μS/cm)",
            min_value=180.0, max_value=750.0,
            value=float(st.session_state.get("Conductivity", default_values["Conductivity"])),
            step=5.0, help="Kemampuan air menghantarkan arus listrik (terkait ion terlarut)."
        )
        organic_carbon = st.number_input(
            "6. Karbon Organik Total / TOC (ppm)",
            min_value=2.0, max_value=30.0,
            value=float(st.session_state.get("Organic_carbon", default_values["Organic_carbon"])),
            step=0.1, help="Total zat organik terlarut yang berasal dari sisa hayati."
        )
        trihalomethanes = st.number_input(
            "7. Trihalometana / THMs (μg/L)",
            min_value=0.5, max_value=130.0,
            value=float(st.session_state.get("Trihalomethanes", default_values["Trihalomethanes"])),
            step=0.5, help="Produk samping klorinasi yang berinteraksi dengan bahan organik."
        )
        turbidity = st.number_input(
            "8. Kekeruhan / Turbidity (NTU)",
            min_value=1.0, max_value=7.0,
            value=float(st.session_state.get("Turbidity", default_values["Turbidity"])),
            step=0.1, help="Tingkat kejernihan/kekeruhan air akibat partikel koloid tersuspensi."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🚀 Estimasi Nilai Derajat Keasaman (pH) Air", type="primary", use_container_width=True)

    # Eksekusi Prediksi
    if btn_predict:
        input_raw = pd.DataFrame([{
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }])

        # Standarisasi dengan Scaler
        if scaler is not None:
            input_scaled = pd.DataFrame(scaler.transform(input_raw), columns=input_raw.columns)
        else:
            input_scaled = input_raw

        # Prediksi Nilai Kontinu pH
        predicted_ph = float(active_model.predict(input_scaled)[0])

        # Batasi rentang display agar tidak melampaui skala wajar air (0 - 14)
        display_ph = max(0.0, min(14.0, predicted_ph))

        # Pengkategorian Kualitas Air
        if display_ph < 6.5:
            box_style = "result-box-acid"
            badge_color = "#e11d48"
            status_title = "AIR BERSIFAT ASAM (DI BAWAH BAKU MUTU)"
            desc_text = (
                "Air memiliki nilai pH di bawah 6.5. Kondisi asam ini dapat menimbulkan rasa asam, "
                "memicu korosi agresif pada pipa logam distribusi air, dan berisiko melarutkan logam berat. "
                "Perlu dilakukan aerasi atau penambahan zat basa (seperti soda abu/kapur) sebelum dikonsumsi."
            )
        elif 6.5 <= display_ph <= 8.5:
            box_style = "result-box-neutral"
            badge_color = "#059669"
            status_title = "AIR NETRAL & IDEAL (SESUAI BAKU MUTU KESEHATAN)"
            desc_text = (
                "Air berada dalam rentang baku mutu ideal air minum menurut <b>Permenkes No. 492/2010</b> dan <b>WHO</b> (pH 6.5 - 8.5). "
                "Sangat aman untuk konsumsi, menjaga keseimbangan metabolisme tubuh, dan tidak menyebabkan korosi maupun kerak pipa."
            )
        else:
            box_style = "result-box-alkali"
            badge_color = "#4f46e5"
            status_title = "AIR BERSIFAT BASA / ALKALI TINGGI"
            desc_text = (
                "Air memiliki nilai pH di atas 8.5. Sifat alkali tinggi cenderung menimbulkan rasa agak pahit/licin (licin sabun) "
                "dan berisiko memicu pengendapan kerak kalsium/mineral pada instalasi pipa dan peralatan pemanas air."
            )

        st.markdown("---")
        st.markdown("### 📋 Hasil Estimasi Nilai pH:")

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            st.markdown(f"""
            <div class="{box_style}" style="text-align: center;">
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #475569;">
                    ESTIMASI pH AIR
                </div>
                <div class="ph-display-number" style="color: {badge_color};">
                    {display_ph:.2f}
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #64748b;">
                    Model: {selected_model_name.split(' (')[0]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="{box_style}">
                <div style="font-size: 0.85rem; font-weight: 800; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
                    ● STATUS: {status_title}
                </div>
                <p style="font-size: 0.95rem; line-height: 1.6; color: #1e293b; margin-bottom: 0.8rem;">
                    {desc_text}
                </p>
                <div style="font-size: 0.8rem; color: #64748b; border-top: 1px dashed rgba(0,0,0,0.15); padding-top: 0.6rem;">
                    <b>Parameter Input:</b> Hardness={hardness} mg/L, Solids={solids} ppm, Chloramines={chloramines} ppm, Sulfate={sulfate} mg/L
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge Bar Visual Interaktif
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📏 Posisi Prediksi pada Skala Indikator pH Meter:")

        pct = max(0, min(100, (display_ph / 14.0) * 100))

        st.markdown(f"""
        <div style="background: linear-gradient(to right, #ef4444 0%, #f97316 25%, #22c55e 50%, #3b82f6 75%, #8b5cf6 100%); height: 26px; border-radius: 13px; position: relative; margin-top: 15px; margin-bottom: 30px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.15);">
            <div style="position: absolute; left: calc({pct:.1f}% - 14px); top: -8px; width: 28px; height: 42px; background: #ffffff; border: 3px solid #0f172a; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800; color: #0f172a;">
                ▼
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #475569; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            <span>0 (Asam Kuat)</span>
            <span>6.5 (Batas Min)</span>
            <span>7.0 (Netral)</span>
            <span>8.5 (Batas Max)</span>
            <span>14 (Basa Kuat)</span>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: KOMPARASI MODEL & METRIK EVALUASI
# ------------------------------------------------------------------------------
with tab_komparasi:
    st.markdown("### 🏆 Hasil Perbandingan 3 Model Regresi (Head-to-Head)")
    st.write(
        "Berdasarkan evaluasi terhadap 557 data uji (*Testing Set*) dengan 5 metrik evaluasi wajib (MAE, MSE, RMSE, MAPE, dan R²):"
    )

    if metrics_df is not None:
        st.dataframe(
            metrics_df.style.highlight_min(subset=["MAE", "MSE", "RMSE", "MAPE (%)"], color="#dcfce7")
                           .highlight_max(subset=["R2 Score"], color="#dcfce7"),
            use_container_width=True
        )
    else:
        st.warning("Tabel hasil komparasi belum ditemukan. Jalankan `compare_regression.py` terlebih dahulu.")

    st.markdown("---")
    st.markdown("#### 🖼️ Grafik Visualisasi Komparasi Menyeluruh:")
    
    comp_img_path = os.path.join(BASE_DIR, "visualisasi", "komparasi_model_regresi.png")
    if os.path.exists(comp_img_path):
        st.image(comp_img_path, caption="Perbandingan 5 Metrik Evaluasi & Tracking Prediksi 40 Sampel Uji", use_container_width=True)
    else:
        st.info("Gambar grafik komparasi belum tersedia di `visualisasi/komparasi_model_regresi.png`.")

    st.markdown("---")
    st.markdown("#### 🌲 Analisis Feature Importance (Tingkat Pengaruh Parameter terhadap pH):")
    rf_img_path = os.path.join(BASE_DIR, "visualisasi", "evaluasi_random_forest.png")
    if os.path.exists(rf_img_path):
        st.image(rf_img_path, caption="Evaluasi Random Forest (Actual vs Pred, Residu, & Feature Importance)", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: PANDUAN BAKU MUTU & TENTANG KELOMPOK
# ------------------------------------------------------------------------------
with tab_edukasi:
    st.markdown("### 📘 Standar Baku Mutu Air Minum (Permenkes & WHO)")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("""
        <div class="input-card" style="border-top: 4px solid #ef4444;">
            <h4 style="color:#e11d48; margin-top:0;">🔴 Air Asam (pH &lt; 6.5)</h4>
            <ul style="font-size:0.9rem; color:#334155; line-height:1.6;">
                <li>Rasa cenderung masam atau getir.</li>
                <li>Memicu korosi logam pada pipa saluran air.</li>
                <li>Dapat menyebabkan iritasi lambung pada orang sensitif.</li>
                <li>Ditemukan di air rawa/gambut atau tercemar limbah asam.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="input-card" style="border-top: 4px solid #10b981;">
            <h4 style="color:#059669; margin-top:0;">🟢 Air Ideal (pH 6.5 - 8.5)</h4>
            <ul style="font-size:0.9rem; color:#334155; line-height:1.6;">
                <li>Standar baku mutu <b>Permenkes No. 492/2010</b>.</li>
                <li>Rasa netral, segar, dan tidak berbau.</li>
                <li>Aman bagi seluruh organ dan saluran pencernaan.</li>
                <li>Keseimbangan mineral kalsium dan magnesium terjaga.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_e3:
        st.markdown("""
        <div class="input-card" style="border-top: 4px solid #6366f1;">
            <h4 style="color:#4f46e5; margin-top:0;">🔵 Air Basa (pH &gt; 8.5)</h4>
            <ul style="font-size:0.9rem; color:#334155; line-height:1.6;">
                <li>Rasa cenderung agak pahit atau bersabun.</li>
                <li>Memicu timbulnya kerak mineral putih pada pipa & ketel.</li>
                <li>Biasanya mengandung kesadahan (kalsium/magnesium) tinggi.</li>
                <li>Dibutuhkan demineralisasi jika kelewat pekat.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 Struktur Pembagian Tugas Proyek Kelompok 3:")
    st.markdown("""
    | Anggota Kelompok | Modul / File Tanggung Jawab | Kontribusi Teknis & Laporan |
    |---|---|---|
    | **Adel** | `preprocessing_regresi.py`<br>`linear_regression_pipeline.py` | Imputasi missing values, outlier IQR clipping, scaling data, data split 80:20, serta implementasi Baseline Model Linear Regression & Persamaan Regresi. |
    | **Juanne** | `random_forest_pipeline.py`<br>`gradient_boosting_pipeline.py`<br>`compare_regression.py` | Implementasi Model Juara Random Forest, Gradient Boosting, komparasi 5 metrik evaluasi lengkap, serta visualisasi grafik analitik. |
    | **Risha** | `app.py`<br>Integrasi UI/UX Web Dashboard | Perancangan arsitektur antarmuka interaktif Streamlit, sistem kalkulasi prediksi, visualisasi indikator pH meter, dan panduan baku mutu air. |
    """)

# Footer Watermark
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small style='color: #94a3b8;'>Tugas Mandiri 11 Machine Learning • Kelompok 3 (Water Potability Regression) © 2026</small></center>",
    unsafe_allow_html=True
)
