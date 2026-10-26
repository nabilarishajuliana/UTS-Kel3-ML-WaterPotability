import os
import sys
import numpy as np
import pandas as pd
import warnings
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Pastikan direktori saat ini ada di sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Prediksi Kelayakan Air Minum",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TEMA & GAYA TAMPILAN: FRESH AQUA / WATER PURITY THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(180deg, #f0fdfa 0%, #e0f2fe 45%, #f8fafc 100%) !important;
        color: #0f172a !important;
    }

    /* Sembunyikan HANYA tombol Deploy, menu 3-titik, dan toolbar actions di sisi kanan */
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stToolbarActions"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* Header dibuat transparan tanpa bar tebal bawaan Streamlit */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background: transparent !important;
        height: 3.5rem !important;
        z-index: 999990 !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
        visibility: visible !important;
        display: flex !important;
    }

    /* Tombol >> (Expand Sidebar) diposisikan FIXED di pojok kiri atas agar SELALU TERLIHAT */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Expand sidebar"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 9999999 !important;
        background: #ffffff !important;
        border: 2px solid #0284c7 !important;
        border-radius: 8px !important;
        color: #0284c7 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        padding: 6px 10px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[aria-label="Expand sidebar"] svg {
        fill: #0284c7 !important;
        stroke: #0284c7 !important;
        color: #0284c7 !important;
        width: 22px !important;
        height: 22px !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    button[aria-label="Expand sidebar"]:hover {
        background: #e0f2fe !important;
        border-color: #0369a1 !important;
        transform: scale(1.08) !important;
    }

    /* Main container max width & spacing */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
    }

    /* Header Nav / Title */
    .aqua-header-card {
        background: #ffffff;
        border: 1px solid #bae6fd;
        border-radius: 14px;
        padding: 1.75rem 2rem;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.08), 0 4px 6px -2px rgba(2, 132, 199, 0.03);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }

    .aqua-header-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #10b981 100%);
    }

    .aqua-badge {
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
        margin-bottom: 0.6rem;
    }

    .aqua-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0c4a6e;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.025em;
    }

    .aqua-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
        line-height: 1.55;
    }

    /* Category Group Cards */
    .param-group-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 15px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
        height: 100%;
    }

    .param-group-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0284c7;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Result Verdict Cards */
    .verdict-card-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.15);
    }

    .verdict-card-unsafe {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(244, 63, 94, 0.15);
    }

    .verdict-tag-safe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #047857;
        text-transform: uppercase;
    }

    .verdict-tag-unsafe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #be123c;
        text-transform: uppercase;
    }

    .verdict-heading {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.4rem 0 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .verdict-subtext {
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    .confidence-meter {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Probability Distribution Box */
    .prob-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04);
        height: 100%;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
        min-width: 350px !important;
        max-width: 380px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    .sidebar-model-active {
        background: #f0fdfa;
        border: 1.5px solid #0d9488;
        border-radius: 8px;
        padding: 0.65rem;
        margin-bottom: 0.45rem;
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.08);
    }

    .sidebar-model-inactive {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem;
        margin-bottom: 0.45rem;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 20px -3px rgba(2, 132, 199, 0.4) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px -4px rgba(2, 132, 199, 0.5) !important;
    }

    /* Input Numbers */
    div[data-testid="stNumberInput"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* Dataframe Table */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Dropdown Selectbox styling to prevent text cutoff */
    div[data-baseweb="select"] {
        width: 100% !important;
    }
    div[data-baseweb="select"] * {
        font-size: 0.84rem !important;
    }
    div[data-baseweb="popover"] {
        max-width: 380px !important;
    }
    div[data-baseweb="popover"] li {
        font-size: 0.84rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        padding: 0.5rem 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)



# ============================================================
# FUNGSI LOAD & TRAIN MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat dataset dan melatih model klasifikasi...")
def load_models():
    """Load dataset, preprocessing, dan latih seluruh model."""
    from preprocessing_p8 import load_and_preprocess

    X_train, X_test, y_train, y_test = load_and_preprocess()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "water_potability.csv")
    df_raw = pd.read_csv(data_path)
    df_raw.fillna(df_raw.median(numeric_only=True), inplace=True)

    feature_cols = [c for c in df_raw.columns if c != "Potability"]
    for col in feature_cols:
        Q1 = df_raw[col].quantile(0.25)
        Q3 = df_raw[col].quantile(0.75)
        IQR = Q3 - Q1
        df_raw[col] = df_raw[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    X_raw = df_raw[feature_cols]
    col_max = X_raw.max()

    # Resampling SMOTE & RUS
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    # 1. SVM (RBF) + RUS (Akurasi Tertinggi: 61.43%)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_rus = SVC(kernel='rbf', probability=True, random_state=42)
        svm_rus.fit(X_train_rus, y_train_rus)

    # 2. SVM (RBF) + SMOTE (F1 Tertinggi: 52.73%)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_smote = SVC(kernel='rbf', probability=True, random_state=42)
        svm_smote.fit(X_train_smote, y_train_smote)

    # 3. Naive Bayes + SMOTE (58.23%)
    nb_smote = GaussianNB()
    nb_smote.fit(X_train_smote, y_train_smote)

    # 4. Decision Tree + SMOTE (Baseline UTS)
    dt_smote = DecisionTreeClassifier(random_state=42)
    dt_smote.fit(X_train_smote, y_train_smote)

    # 5. Naive Bayes + RUS
    nb_rus = GaussianNB()
    nb_rus.fit(X_train_rus, y_train_rus)

    # 6. Decision Tree + RUS
    dt_rus = DecisionTreeClassifier(random_state=42)
    dt_rus.fit(X_train_rus, y_train_rus)

    # Model Original (Tanpa Resampling)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_ori = SVC(kernel='rbf', probability=True, random_state=42)
        svm_ori.fit(X_train, y_train)

    nb_ori = GaussianNB()
    nb_ori.fit(X_train, y_train)

    dt_ori = DecisionTreeClassifier(random_state=42)
    dt_ori.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    models_info = {
        "SVM (RBF) - Original (Tanpa Resampling)": {
            "model": svm_ori,
            "badge": "Akurasi Mentah Puncak (65.85%)",
            "accuracy": accuracy_score(y_test, svm_ori.predict(X_test)),
            "precision": precision_score(y_test, svm_ori.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, svm_ori.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, svm_ori.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, svm_ori.predict(X_test)),
            "type": "Support Vector Machine",
            "sampling": "Original (Tanpa Resampling)"
        },
        "SVM (RBF) - RUS": {
            "model": svm_rus,
            "badge": "Akurasi Resampling Tertinggi (61.43%)",
            "accuracy": accuracy_score(y_test, svm_rus.predict(X_test)),
            "precision": precision_score(y_test, svm_rus.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, svm_rus.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, svm_rus.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, svm_rus.predict(X_test)),
            "type": "Support Vector Machine",
            "sampling": "Random Under Sampling (RUS)"
        },
        "SVM (RBF) - SMOTE": {
            "model": svm_smote,
            "badge": "F1 Tertinggi (52.73%)",
            "accuracy": accuracy_score(y_test, svm_smote.predict(X_test)),
            "precision": precision_score(y_test, svm_smote.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, svm_smote.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, svm_smote.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, svm_smote.predict(X_test)),
            "type": "Support Vector Machine",
            "sampling": "SMOTE (Oversampling)"
        },
        "Naive Bayes - Original": {
            "model": nb_ori,
            "badge": "Original Tanpa Resampling (60.52%)",
            "accuracy": accuracy_score(y_test, nb_ori.predict(X_test)),
            "precision": precision_score(y_test, nb_ori.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, nb_ori.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, nb_ori.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, nb_ori.predict(X_test)),
            "type": "Gaussian Naive Bayes",
            "sampling": "Original (Tanpa Resampling)"
        },
        "Naive Bayes - SMOTE": {
            "model": nb_smote,
            "badge": "F1 Naive Bayes Tertinggi (50.00%)",
            "accuracy": accuracy_score(y_test, nb_smote.predict(X_test)),
            "precision": precision_score(y_test, nb_smote.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, nb_smote.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, nb_smote.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, nb_smote.predict(X_test)),
            "type": "Gaussian Naive Bayes",
            "sampling": "SMOTE (Oversampling)"
        },
        "Naive Bayes - RUS": {
            "model": nb_rus,
            "badge": "RUS Undersampling (57.62%)",
            "accuracy": accuracy_score(y_test, nb_rus.predict(X_test)),
            "precision": precision_score(y_test, nb_rus.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, nb_rus.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, nb_rus.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, nb_rus.predict(X_test)),
            "type": "Gaussian Naive Bayes",
            "sampling": "Random Under Sampling (RUS)"
        },
        "Decision Tree - Original": {
            "model": dt_ori,
            "badge": "DT Original (58.08%)",
            "accuracy": accuracy_score(y_test, dt_ori.predict(X_test)),
            "precision": precision_score(y_test, dt_ori.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, dt_ori.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, dt_ori.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, dt_ori.predict(X_test)),
            "type": "Decision Tree",
            "sampling": "Original (Tanpa Resampling)"
        },
        "Decision Tree - UTS (SMOTE)": {
            "model": dt_smote,
            "badge": "Baseline UTS (56.40%)",
            "accuracy": accuracy_score(y_test, dt_smote.predict(X_test)),
            "precision": precision_score(y_test, dt_smote.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, dt_smote.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, dt_smote.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, dt_smote.predict(X_test)),
            "type": "Decision Tree",
            "sampling": "SMOTE (Oversampling)"
        },
        "Decision Tree - RUS": {
            "model": dt_rus,
            "badge": "DT RUS (55.03%)",
            "accuracy": accuracy_score(y_test, dt_rus.predict(X_test)),
            "precision": precision_score(y_test, dt_rus.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, dt_rus.predict(X_test), zero_division=0),
            "f1": f1_score(y_test, dt_rus.predict(X_test), zero_division=0),
            "cm": confusion_matrix(y_test, dt_rus.predict(X_test)),
            "type": "Decision Tree",
            "sampling": "Random Under Sampling (RUS)"
        },
    }

    return models_info, col_max, feature_cols


# ============================================================
# PARAMETER ACUAN LABORATORIUM (STANDAR WHO)
# ============================================================
PARAM_INFO = {
    # Grup 1: Sifat Fisika & Asam Basa
    "ph": {
        "group": "fisika",
        "label": "Tingkat Keasaman (pH)",
        "min": 0.0, "max": 14.0, "default": 7.08, "step": 0.01, "unit": "",
        "threshold": "6.5 - 8.5", "desc": "Baku mutu air minum: pH netral (6.5 - 8.5)."
    },
    "Hardness": {
        "group": "fisika",
        "label": "Kesadahan Air (Hardness)",
        "min": 47.0, "max": 323.0, "default": 196.3, "step": 0.1, "unit": "mg/L",
        "threshold": "< 200 mg/L", "desc": "Kadar kalsium dan magnesium terlarut."
    },
    "Turbidity": {
        "group": "fisika",
        "label": "Kekeruhan Air (Turbidity)",
        "min": 1.45, "max": 6.74, "default": 3.97, "step": 0.01, "unit": "NTU",
        "threshold": "< 5.0 NTU", "desc": "Tingkat kekeruhan partikel suspensi."
    },

    # Grup 2: Zat Terlarut & Konduktivitas
    "Solids": {
        "group": "kimia_mineral",
        "label": "Padatan Terlarut (TDS)",
        "min": 320.0, "max": 61227.0, "default": 20927.0, "step": 1.0, "unit": "ppm",
        "threshold": "< 500 - 1000 ppm", "desc": "Total zat padat terlarut (mineral, garam)."
    },
    "Sulfate": {
        "group": "kimia_mineral",
        "label": "Kandungan Sulfat (Sulfate)",
        "min": 129.0, "max": 481.0, "default": 333.0, "step": 0.1, "unit": "mg/L",
        "threshold": "< 250 mg/L", "desc": "Senyawa sulfat alami dalam batuan/tanah."
    },
    "Conductivity": {
        "group": "kimia_mineral",
        "label": "Daya Hantar Listrik",
        "min": 181.0, "max": 753.0, "default": 426.6, "step": 0.1, "unit": "μS/cm",
        "threshold": "< 400 μS/cm", "desc": "Kemampuan larutan menghantarkan arus listrik."
    },

    # Grup 3: Disinfektan & Karbon Organik
    "Chloramines": {
        "group": "disinfektan",
        "label": "Kadar Kloramin (Chloramines)",
        "min": 0.35, "max": 13.13, "default": 7.13, "step": 0.01, "unit": "ppm",
        "threshold": "< 4.0 ppm", "desc": "Zat disinfektan untuk sterilisasi kuman."
    },
    "Organic_carbon": {
        "group": "disinfektan",
        "label": "Karbon Organik (Total Organic)",
        "min": 2.2, "max": 28.3, "default": 14.28, "step": 0.01, "unit": "ppm",
        "threshold": "< 2.0 - 4.0 ppm", "desc": "Kandungan bahan organik terurai."
    },
    "Trihalomethanes": {
        "group": "disinfektan",
        "label": "Trihalometana (THMs)",
        "min": 0.738, "max": 124.0, "default": 66.4, "step": 0.01, "unit": "μg/L",
        "threshold": "< 80.0 μg/L", "desc": "Senyawa sampingan hasil klorinasi air."
    },
}


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="aqua-header-card">
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1.5rem;">
        <!-- Sisi Kiri: Judul & Informasi Proyek -->
        <div style="flex: 1.3; min-width: 320px;">
            <div class="aqua-badge">LABORATORIUM MUTU AIR &bull; MODUL POTABILITAS</div>
            <h1 class="aqua-title">Sistem Prediksi Kelayakan Air Minum</h1>
            <p class="aqua-subtitle">
                Platform pengujian kelayakan air konsumsi berbasis machine learning dengan analisis 9 parameter fisik-kimia.
                Inferensi ditenagai model terbaik: <b>Support Vector Machine (Kernel RBF)</b>.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #0369a1; background: #e0f2fe; padding: 0.25rem 0.6rem; border-radius: 4px; border: 1px solid #bae6fd;">3.276 Sampel Teruji</span>
                <span style="font-size: 0.76rem; font-weight: 700; color: #0f766e; background: #ccfbf1; padding: 0.25rem 0.6rem; border-radius: 4px; border: 1px solid #99f6e4;">Standar: WHO &amp; Kemenkes</span>
                <span style="font-size: 0.76rem; font-weight: 700; color: #334155; background: #f1f5f9; padding: 0.25rem 0.6rem; border-radius: 4px; border: 1px solid #e2e8f0;">Split: 80% Train : 20% Test</span>
            </div>
        </div>
        <!-- Sisi Kanan: 3 Kartu Ringkasan Benchmark KPI -->
        <div style="display: flex; gap: 0.85rem; flex-wrap: wrap;">
            <div style="background: linear-gradient(135deg, #f0fdfa 0%, #e0f2fe 100%); border: 1px solid #7dd3fc; border-radius: 10px; padding: 0.9rem 1.15rem; min-width: 130px; text-align: center; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.08);">
                <div style="font-size: 0.7rem; font-weight: 700; color: #0369a1; text-transform: uppercase; letter-spacing: 0.05em;">Akurasi Puncak</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #0c4a6e; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0;">61.43%</div>
                <div style="font-size: 0.72rem; color: #0f766e; font-weight: 700;">SVM RBF + RUS</div>
            </div>
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 1px solid #86efac; border-radius: 10px; padding: 0.9rem 1.15rem; min-width: 130px; text-align: center; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);">
                <div style="font-size: 0.7rem; font-weight: 700; color: #15803d; text-transform: uppercase; letter-spacing: 0.05em;">F1 Terbaik</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #065f46; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0;">52.73%</div>
                <div style="font-size: 0.72rem; color: #166534; font-weight: 700;">SVM RBF + SMOTE</div>
            </div>
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.9rem 1.15rem; min-width: 130px; text-align: center; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);">
                <div style="font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">Spesifisitas</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #1e293b; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0;">71.00%</div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">Tolak Air Bahaya</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MEMUAT MODEL
# ============================================================
try:
    models_info, col_max, feature_cols = load_models()
except Exception as e:
    st.error(f"Gagal memuat model klasifikasi: {e}")
    st.stop()


# ============================================================
# SIDEBAR KONFIGURASI
# ============================================================
with st.sidebar:
    st.markdown("<h3 style='color:#0c4a6e; font-size:1.05rem; font-weight:800; margin:0 0 0.1rem 0; letter-spacing:-0.01em;'>PENGATURAN MODEL</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.74rem; color:#64748b; margin:0 0 0.35rem 0;'>Pilih model inferensi & evaluasi 9 konfigurasi.</p>", unsafe_allow_html=True)

    model_names = list(models_info.keys())
    selected_model = st.selectbox(
        "Model Inferensi:",
        model_names,
        index=0,
        label_visibility="collapsed"
    )

    info_curr = models_info[selected_model]
    badge_curr = info_curr.get('badge', '')

    # Kartu Detail Model Terpilih (Ultra Compact)
    card_html = (
        f'<div style="background: linear-gradient(135deg, #f0fdfa 0%, #e6fffa 100%); border: 1.5px solid #0d9488; border-radius: 8px; padding: 0.45rem 0.65rem; margin-top: 0.2rem; margin-bottom: 0.45rem; box-shadow: 0 2px 8px -2px rgba(13, 148, 136, 0.1);">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">'
        f'<span style="font-size:0.68rem; font-weight:800; color:#0f766e; text-transform:uppercase; letter-spacing:0.04em;">[TERPILIH]</span>'
        f'<span style="font-size:0.67rem; font-weight:700; color:#0284c7; background:#e0f2fe; padding:0.1rem 0.4rem; border-radius:4px;">{badge_curr}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem; font-weight:800; color:#0f172a; margin-bottom:0.25rem; line-height:1.2;">{selected_model}</div>'
        f'<div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.2rem; font-family:monospace; font-size:0.69rem; color:#334155; background:#ffffff; padding:0.3rem 0.45rem; border-radius:5px; border:1px solid #ccfbf1;">'
        f'<div>Akurasi : <b style="color:#0f766e;">{info_curr["accuracy"]*100:.2f}%</b></div>'
        f'<div>F1-Score: <b style="color:#0f766e;">{info_curr["f1"]*100:.2f}%</b></div>'
        f'<div>Recall  : <b>{info_curr["recall"]*100:.2f}%</b></div>'
        f'<div>Presisi : <b>{info_curr["precision"]*100:.2f}%</b></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Leaderboard 9 Model Ringkas (Muat 1 Layar / Siap SS)
    st.markdown("<h4 style='color:#0c4a6e; font-size:0.80rem; font-weight:800; margin:0 0 0.25rem 0; letter-spacing:-0.01em;'>LEADERBOARD 9 MODEL (SIAP SS)</h4>", unsafe_allow_html=True)

    table_rows = ""
    for idx, (name, info) in enumerate(models_info.items()):
        is_sel = (name == selected_model)
        row_bg = "#ccfbf1" if is_sel else ("#f8fafc" if idx % 2 == 0 else "#ffffff")
        text_color = "color:#0f766e; font-weight:700;" if is_sel else "color:#334155;"
        indicator = "👉 " if is_sel else ""

        short_name = (name.replace(" (Tanpa Resampling)", "")
                           .replace("Decision Tree (Baseline UTS)", "DT Baseline")
                           .replace("Decision Tree - UTS (SMOTE)", "DT SMOTE")
                           .replace("Decision Tree - RUS", "DT RUS")
                           .replace("Decision Tree - Original", "DT Original")
                           .replace("Support Vector Machine (Kernel RBF)", "SVM RBF")
                           .replace("Gaussian Naive Bayes", "Naive Bayes"))

        table_rows += (
            f'<tr style="background:{row_bg}; border-bottom:1px solid #e2e8f0; font-size:0.67rem; line-height:1.1;">'
            f'<td style="padding:0.16rem 0.3rem; {text_color}">{indicator}{short_name}</td>'
            f'<td style="padding:0.16rem 0.2rem; text-align:center; font-family:monospace; {text_color}">{info["accuracy"]*100:.1f}%</td>'
            f'<td style="padding:0.16rem 0.2rem; text-align:center; font-family:monospace; {text_color}">{info["f1"]*100:.1f}%</td>'
            f'</tr>'
        )

    table_html = (
        '<div style="border:1px solid #cbd5e1; border-radius:6px; overflow:hidden; margin-bottom:0.4rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead>'
        '<tr style="background:#e2e8f0; font-size:0.68rem; font-weight:700; color:#1e293b;">'
        '<th style="padding:0.22rem 0.3rem; text-align:left;">Model &amp; Skenario</th>'
        '<th style="padding:0.22rem 0.2rem; text-align:center;">Acc</th>'
        '<th style="padding:0.22rem 0.2rem; text-align:center;">F1</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{table_rows}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    with st.expander("Lihat Rincian Kartu Semua Model"):
        for name, info in models_info.items():
            is_active = (name == selected_model)
            css_class = "sidebar-model-active" if is_active else "sidebar-model-inactive"
            status_tag = "<span style='color:#0d9488; font-weight:700;'>[TERPILIH]</span> " if is_active else ""
            card_item = (
                f'<div class="{css_class}" style="padding:0.35rem 0.45rem; margin-bottom:0.3rem;">'
                f'<div style="font-size:0.72rem; font-weight:700; color:{"#0f766e" if is_active else "#334155"};">'
                f'{status_tag}{name}'
                f'</div>'
                f'<div style="font-family:monospace; font-size:0.66rem; color:#64748b; margin-top:0.1rem; line-height:1.25;">'
                f'Acc: <b style="color:{"#0f766e" if is_active else "#0f172a"};">{info["accuracy"]*100:.2f}%</b> &bull; F1: <b>{info["f1"]*100:.2f}%</b> &bull; Rec: <b>{info["recall"]*100:.2f}%</b>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_item, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.68rem; color:#64748b; line-height:1.35; border-top:1px solid #e2e8f0; padding-top:0.35rem; margin-top:0.3rem;">
        <b>Dataset:</b> Water Potability &bull; 656 Data Uji<br>
        <b>Kelompok 3:</b> Adel, Juanne, Risha &bull; D4 TI UNAIR
    </div>
    """, unsafe_allow_html=True)



# ============================================================
# TOMBOL PRESET SAMPEL
# ============================================================
st.markdown("<h3 style='color:#0c4a6e; font-size:1.15rem; font-weight:700; margin-bottom:0.2rem;'>PENGISIAN PARAMETER SAMPEL AIR</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.88rem; color:#64748b; margin-bottom:1rem;'>Gunakan tombol preset pengujian di bawah ini atau sesuaikan angka pada masing-masing kartu parameter.</p>", unsafe_allow_html=True)

p_col1, p_col2, p_col3, p_spacer = st.columns([1.8, 1.8, 1.8, 2.5])
preset_clicked = None
with p_col1:
    if st.button("Standar Air Sehat (WHO)", help="Muat parameter ideal yang aman dikonsumsi"):
        preset_clicked = "safe"
with p_col2:
    if st.button("Sampel Air Tercemar", help="Muat parameter ekstrem yang membahayakan"):
        preset_clicked = "unsafe"
with p_col3:
    if st.button("Reset Rerata Standar", help="Kembalikan ke nilai rata-rata dataset"):
        preset_clicked = "default"

# Nilai preset
preset_values = {
    "safe": {
        "ph": 7.35, "Hardness": 170.0, "Solids": 850.0,
        "Chloramines": 2.50, "Sulfate": 210.0, "Conductivity": 320.0,
        "Organic_carbon": 2.80, "Trihalomethanes": 45.0, "Turbidity": 2.10
    },
    "unsafe": {
        "ph": 4.10, "Hardness": 285.0, "Solids": 35000.0,
        "Chloramines": 9.80, "Sulfate": 420.0, "Conductivity": 680.0,
        "Organic_carbon": 19.5, "Trihalomethanes": 98.0, "Turbidity": 6.20
    }
}


# ============================================================
# FORM PARAMETER DALAM 3 KELOMPOK
# ============================================================
card_col1, card_col2, card_col3 = st.columns(3)
input_values = {}

# Kelompok 1: Fisika & Asam Basa
with card_col1:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#0284c7; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem; border-bottom:1px solid #f1f5f9; padding-bottom:0.4rem;'>1. SIFAT FISIKA & KEASAMAN</div>", unsafe_allow_html=True)

        for k in ["ph", "Hardness", "Turbidity"]:
            info = PARAM_INFO[k]
            def_val = float(info['default'])
            if preset_clicked in preset_values:
                def_val = float(preset_values[preset_clicked][k])

            unit_label = f" ({info['unit']})" if info['unit'] else ""
            val = st.number_input(
                f"{info['label']}{unit_label}",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=def_val,
                step=float(info['step']),
                key=f"in_{k}",
                help=f"Ambang Batas: {info['threshold']}. {info['desc']}"
            )
            input_values[k] = val

# Kelompok 2: Zat Padat & Kimiawi
with card_col2:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#0284c7; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem; border-bottom:1px solid #f1f5f9; padding-bottom:0.4rem;'>2. PADATAN & DAYA HANTAR</div>", unsafe_allow_html=True)

        for k in ["Solids", "Sulfate", "Conductivity"]:
            info = PARAM_INFO[k]
            def_val = float(info['default'])
            if preset_clicked in preset_values:
                def_val = float(preset_values[preset_clicked][k])

            unit_label = f" ({info['unit']})" if info['unit'] else ""
            val = st.number_input(
                f"{info['label']}{unit_label}",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=def_val,
                step=float(info['step']),
                key=f"in_{k}",
                help=f"Ambang Batas: {info['threshold']}. {info['desc']}"
            )
            input_values[k] = val

# Kelompok 3: Disinfektan & Organik
with card_col3:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.85rem; font-weight:800; color:#0284c7; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem; border-bottom:1px solid #f1f5f9; padding-bottom:0.4rem;'>3. DISINFEKTAN & SENYAWA ORGANIK</div>", unsafe_allow_html=True)

        for k in ["Chloramines", "Organic_carbon", "Trihalomethanes"]:
            info = PARAM_INFO[k]
            def_val = float(info['default'])
            if preset_clicked in preset_values:
                def_val = float(preset_values[preset_clicked][k])

            unit_label = f" ({info['unit']})" if info['unit'] else ""
            val = st.number_input(
                f"{info['label']}{unit_label}",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=def_val,
                step=float(info['step']),
                key=f"in_{k}",
                help=f"Ambang Batas: {info['threshold']}. {info['desc']}"
            )
            input_values[k] = val



st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# Tombol Eksekusi
btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2.5, 1.5])
with btn_col2:
    predict_btn = st.button("JALANKAN INFERENSI KELAYAKAN AIR", use_container_width=True, type="primary")


# ============================================================
# HASIL ANALISIS
# ============================================================
if predict_btn:
    # 1. Transformasi Input
    input_arr = np.array([[input_values[k] for k in feature_cols]])
    input_scaled = input_arr / col_max.values.reshape(1, -1)

    # 2. Inferensi Model
    chosen_model = models_info[selected_model]['model']
    prediction = int(chosen_model.predict(input_scaled)[0])
    proba = chosen_model.predict_proba(input_scaled)[0]

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#0c4a6e; font-size:1.25rem; font-weight:800; margin-bottom:1rem;'>HASIL EVALUASI LABORATORIUM</h3>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.1, 1.0])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="verdict-card-safe">
                <div class="verdict-tag-safe">[ HASIL PENGUJIAN: AIR POTABLE ]</div>
                <div class="verdict-heading" style="color:#065f46;">LAYAK DIKONSUMSI</div>
                <div class="verdict-subtext" style="color:#166534;">
                    Sampel air memenuhi kriteria baku mutu air minum. Berdasarkan pemetaan ruang fitur model, konsentrasi mineral dan disinfektan berada pada rentang toleransi yang aman bagi tubuh manusia.
                </div>
                <div style="font-size:0.8rem; font-weight:600; color:#047857; text-transform:uppercase; letter-spacing:0.04em;">Tingkat Keyakinan Model:</div>
                <div class="confidence-meter" style="color:#059669;">{proba[1]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card-unsafe">
                <div class="verdict-tag-unsafe">[ HASIL PENGUJIAN: AIR NON-POTABLE ]</div>
                <div class="verdict-heading" style="color:#9f1239;">TIDAK LAYAK DIKONSUMSI</div>
                <div class="verdict-subtext" style="color:#881337;">
                    Sampel air terindikasi tidak memenuhi standar keamanan air minum. Terdapat parameter yang berpotensi membahayakan kesehatan atau memiliki tingkat keasaman, padatan, maupun zat kimia di luar ambang batas wajar.
                </div>
                <div style="font-size:0.8rem; font-weight:600; color:#be123c; text-transform:uppercase; letter-spacing:0.04em;">Tingkat Keyakinan Model:</div>
                <div class="confidence-meter" style="color:#e11d48;">{proba[0]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="prob-box">
            <div style="font-size:0.85rem; font-weight:700; color:#0c4a6e; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1.2rem;">
                DISTRIBUSI PROBABILITAS KELAS
            </div>
            <div style="margin-bottom:1.2rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; margin-bottom:0.4rem;">
                    <span style="font-weight:600; color:#475569;">Kelas 0 &bull; Tidak Layak Minum</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-weight:700; color:#e11d48;">{proba[0]*100:.2f}%</span>
                </div>
                <div style="background:#f1f5f9; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="background:#f43f5e; width:{proba[0]*100:.2f}%; height:10px; border-radius:6px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; margin-bottom:0.4rem;">
                    <span style="font-weight:600; color:#475569;">Kelas 1 &bull; Layak Minum</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-weight:700; color:#059669;">{proba[1]*100:.2f}%</span>
                </div>
                <div style="background:#f1f5f9; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="background:#10b981; width:{proba[1]*100:.2f}%; height:10px; border-radius:6px;"></div>
                </div>
            </div>
            <div style="margin-top:1.5rem; padding-top:0.8rem; border-top:1px solid #f1f5f9; font-size:0.78rem; color:#64748b;">
                Model Aktif: <b style="color:#0c4a6e;">{selected_model}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tabel Evaluasi Detail Parameter terhadap Standar Baku Mutu WHO
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#0c4a6e; font-size:1.05rem; font-weight:700;'>KOMPARASI PARAMETER SAMPEL DENGAN STANDAR RUJUKAN</h4>", unsafe_allow_html=True)

    recap_rows = []
    for k in feature_cols:
        val = input_values[k]
        unit = PARAM_INFO[k]["unit"]
        unit_str = f" {unit}" if unit else ""

        # Verifikasi status batas aman acuan
        status = "Normal / Sesuai Acuan"
        if k == "ph" and (val < 6.5 or val > 8.5):
            status = "Peringatan: Melebihi Toleransi"
        elif k == "Chloramines" and val > 4.0:
            status = "Peringatan: Kloramin Tinggi"
        elif k == "Sulfate" and val > 250.0:
            status = "Peringatan: Sulfat Tinggi"
        elif k == "Trihalomethanes" and val > 80.0:
            status = "Peringatan: THM Tinggi"
        elif k == "Turbidity" and val > 5.0:
            status = "Peringatan: Kekeruhan Tinggi"

        recap_rows.append({
            "Parameter Uji": PARAM_INFO[k]["label"],
            "Nilai Sampel": f"{val:.2f}{unit_str}",
            "Standar Rujukan (WHO / Kemenkes)": PARAM_INFO[k]["threshold"],
            "Status Kesesuaian": status,
            "Peranan Fisik-Kimia": PARAM_INFO[k]["desc"]
        })

    df_recap = pd.DataFrame(recap_rows)
    st.dataframe(df_recap, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("<div style='margin-top:3.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #cbd5e1; padding:1.5rem 0; text-align:center; font-size:0.8rem; color:#64748b;">
    Sistem Klasifikasi Kualitas Air Minum &bull; Praktikum Machine Learning Pertemuan 8 &bull; Kelompok 3<br>
    Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga Surabaya
</div>
""", unsafe_allow_html=True)
