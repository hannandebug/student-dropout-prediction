import os
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tensorflow import keras

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_SCRIPT_DIR, '..', 'model')
_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', 'data')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LANGUAGE DICTIONARY
# ============================================================
LANG = {
    "ID": {
        "badge": "GDGoC USU — Machine Learning Pathway",
        "title": "Prediksi Risiko Dropout Mahasiswa",
        "subtitle": "Masukkan data akademik mahasiswa untuk memprediksi risiko dropout menggunakan Machine Learning",
        "nav_predict": "🔍 Prediksi",
        "nav_model": "📊 Info Model",
        "metric_data": "📂 Total Data",
        "metric_data_sub": "mahasiswa",
        "metric_acc": "🏆 Accuracy Terbaik",
        "metric_f1": "🎯 F1-Score Terbaik",
        "metric_models": "🤖 Jumlah Model",
        "section_input": "📋 DATA MAHASISWA",
        "section_personal": "👤 Data Personal",
        "age_label": "🎂 Usia saat Mendaftar",
        "gender_label": "⚧️ Jenis Kelamin",
        "gender_opts": ["Perempuan", "Laki-laki"],
        "gender_vals": ["Perempuan", "Laki-laki"],
        "displaced_label": "🏠 Mahasiswa Perantau?",
        "section_finance": "💰 Data Finansial",
        "scholarship_label": "🎓 Penerima Beasiswa?",
        "debtor_label": "💳 Memiliki Tunggakan?",
        "tuition_label": "✅ UKT Lancar?",
        "yes_no": ["Tidak", "Ya"],
        "yes_no_vals": ["Tidak", "Ya"],
        "yes_no_rev": ["Ya", "Tidak"],
        "yes_no_rev_vals": ["Ya", "Tidak"],
        "section_academic": "📚 Data Akademik",
        "ipk_masuk_label": "📊 IPK Masuk (0.00 - 4.00)",
        "sem1_header": "📅 Semester 1",
        "sem2_header": "📅 Semester 2",
        "sks_ambil": "📖 SKS Diambil",
        "sks_lulus": "✔️ SKS Lulus",
        "ipk_sem": "⭐ IPK Semester",
        "predict_btn": "Prediksi Sekarang",
        "result_section": "📈 Hasil Prediksi",
        "main_pred_label": "Prediksi Utama",
        "based_on": "Berdasarkan Random Forest · F1 Macro 70.57%",
        "risk_title": "⚠️ Risiko Dropout",
        "risk_low": "✅ Tingkat risiko rendah",
        "risk_mid": "⚠️ Tingkat risiko sedang",
        "risk_high": "🚨 Tingkat risiko tinggi",
        "comparison_title": "🔄 Perbandingan Model",
        "main_badge": "utama",
        "rec_title": "💡 Rekomendasi",
        "rec_dropout": """**🚨 Mahasiswa ini berisiko tinggi untuk Dropout.**\n\nLangkah yang disarankan:\n- 👨‍🏫 Segera konsultasikan dengan dosen pembimbing akademik\n- 💰 Cek apakah ada kendala finansial yang bisa diselesaikan lewat beasiswa\n- 📚 Evaluasi beban SKS — pertimbangkan untuk mengurangi jumlah SKS\n- 🤝 Pastikan mahasiswa mendapat dukungan akademik yang cukup""",
        "rec_enrolled": """**⚠️ Mahasiswa masih aktif kuliah.**\n\nPantauan yang disarankan:\n- 📊 Monitor perkembangan IPK setiap semester\n- 📖 Perhatikan tren SKS yang diambil vs yang lulus\n- 🏛️ Dorong mahasiswa untuk memanfaatkan fasilitas bimbingan akademik""",
        "rec_graduate": """**🎓 Mahasiswa diprediksi akan Lulus.**\n\nPertahankan kondisi ini dengan:\n- ⭐ Jaga konsistensi IPK setiap semester\n- ✅ Pastikan pembayaran UKT tetap lancar\n- 💼 Manfaatkan program magang atau kegiatan penunjang karier""",
        "disclaimer": "⚠️ Prediksi bersifat indikatif · Dataset: UCI ML Repository (4.424 data) · Hannan Rava Mahardika",
        "info_title": "ℹ️ Informasi Model & Dataset",
        "info_dataset_title": "📂 Dataset",
        "info_dataset_body": """- **📌 Sumber:** UCI ML Repository — Student Dropout and Academic Success\n- **👥 Jumlah data:** 4.424 mahasiswa\n- **🔢 Jumlah fitur:** 36 fitur akademik, demografis, dan ekonomi\n- **🎯 Target:** Dropout / Masih Aktif / Lulus""",
        "info_model_title": "🤖 Model yang Digunakan",
        "info_feature_title": "🔍 Fitur Paling Berpengaruh",
        "info_feature_body": """Berdasarkan analisis Random Forest, faktor yang paling mempengaruhi risiko dropout:\n1. 📖 Approval rate SKS Semester 2 (fitur engineering)\n2. 📖 Jumlah SKS yang lulus di Semester 2\n3. 📖 Approval rate SKS Semester 1 (fitur engineering)\n4. ⭐ IPK Semester 2\n5. 📖 Jumlah SKS yang lulus di Semester 1\n6. ⭐ IPK Semester 1\n7. 📊 Nilai masuk\n8. 📈 Tren nilai antar semester (fitur engineering)\n9. 🎂 Usia saat mendaftar\n10. 📜 Nilai kualifikasi sebelumnya""",
        "info_note_title": "📝 Catatan Penting",
        "info_note_body": """- 🌍 Model dilatih menggunakan data mahasiswa dari institusi internasional (Portugal)\n- 📊 Input IPK dikonversi secara proporsional ke skala dataset\n- 💹 Kondisi ekonomi menggunakan nilai median dataset\n- ⚠️ Prediksi bersifat indikatif, bukan keputusan final""",
        "info_built": "👨‍💻 Dibuat oleh",
        "info_stack": "🛠️ Tech Stack",
        "info_star_note": "⭐ = Model utama yang digunakan",
        "error_sks1": "❌ SKS lulus tidak boleh lebih besar dari SKS yang diambil (Semester 1)",
        "error_sks2": "❌ SKS lulus tidak boleh lebih besar dari SKS yang diambil (Semester 2)",
        "proba_title": "📊 Probabilitas Prediksi",
        "proba_sub": "Neural Network",
        "labels": ["Dropout", "Masih Aktif", "Lulus"],
        "placeholder": "📝 Isi data mahasiswa di form, lalu klik 🚀 Prediksi Sekarang",
        "processing": "⏳ Memproses...",
    },
    "EN": {
        "badge": "GDGoC USU — Machine Learning Pathway",
        "title": "Student Dropout Risk Prediction",
        "subtitle": "Enter student academic data to predict dropout risk using Machine Learning",
        "nav_predict": "🔍 Prediction",
        "nav_model": "📊 Model Info",
        "metric_data": "📂 Total Data",
        "metric_data_sub": "students",
        "metric_acc": "🏆 Best Accuracy",
        "metric_f1": "🎯 Best F1-Score",
        "metric_models": "🤖 Total Models",
        "section_input": "📋 STUDENT DATA",
        "section_personal": "👤 Personal Data",
        "age_label": "🎂 Age at Enrollment",
        "gender_label": "⚧️ Gender",
        "gender_opts": ["Female", "Male"],
        "gender_vals": ["Female", "Male"],
        "displaced_label": "🏠 Displaced Student?",
        "section_finance": "💰 Financial Data",
        "scholarship_label": "🎓 Scholarship Holder?",
        "debtor_label": "💳 Has Tuition Debt?",
        "tuition_label": "✅ Tuition Up to Date?",
        "yes_no": ["No", "Yes"],
        "yes_no_vals": ["No", "Yes"],
        "yes_no_rev": ["Yes", "No"],
        "yes_no_rev_vals": ["Yes", "No"],
        "section_academic": "📚 Academic Data",
        "ipk_masuk_label": "📊 Admission GPA (0.00 - 4.00)",
        "sem1_header": "📅 Semester 1",
        "sem2_header": "📅 Semester 2",
        "sks_ambil": "📖 Credits Enrolled",
        "sks_lulus": "✔️ Credits Approved",
        "ipk_sem": "⭐ Semester GPA",
        "predict_btn": "Predict Now",
        "result_section": "📈 Prediction Result",
        "main_pred_label": "Main Prediction",
        "based_on": "Based on Random Forest · F1 Macro 70.57%",
        "risk_title": "⚠️ Dropout Risk",
        "risk_low": "✅ Low risk level",
        "risk_mid": "⚠️ Medium risk level",
        "risk_high": "🚨 High risk level",
        "comparison_title": "🔄 Model Comparison",
        "main_badge": "main",
        "rec_title": "💡 Recommendation",
        "rec_dropout": """**🚨 This student has a high risk of Dropout.**\n\nSuggested actions:\n- 👨‍🏫 Consult with academic advisor immediately\n- 💰 Check for financial issues that can be resolved through scholarships\n- 📚 Evaluate credit load — consider reducing credits\n- 🤝 Ensure sufficient academic support""",
        "rec_enrolled": """**⚠️ Student is still actively enrolled.**\n\nSuggested monitoring:\n- 📊 Track GPA development each semester\n- 📖 Monitor enrolled vs approved credits trend\n- 🏛️ Encourage student to use academic counseling facilities""",
        "rec_graduate": """**🎓 Student is predicted to Graduate.**\n\nMaintain this condition by:\n- ⭐ Keep GPA consistent each semester\n- ✅ Ensure tuition payments remain on time\n- 💼 Take advantage of internship or career development programs""",
        "disclaimer": "⚠️ Predictions are indicative · Dataset: UCI ML Repository (4,424 records) · Hannan Rava Mahardika",
        "info_title": "ℹ️ Model & Dataset Information",
        "info_dataset_title": "📂 Dataset",
        "info_dataset_body": """- **📌 Source:** UCI ML Repository — Student Dropout and Academic Success\n- **👥 Total records:** 4,424 students\n- **🔢 Features:** 36 academic, demographic, and economic features\n- **🎯 Target:** Dropout / Enrolled / Graduate""",
        "info_model_title": "🤖 Models Used",
        "info_feature_title": "🔍 Most Influential Features",
        "info_feature_body": """Based on Random Forest feature importance analysis:\n1. 📖 Semester 2 SKS approval rate (engineered)\n2. 📖 Credits approved in Semester 2\n3. 📖 Semester 1 SKS approval rate (engineered)\n4. ⭐ Semester 2 GPA\n5. 📖 Credits approved in Semester 1\n6. ⭐ Semester 1 GPA\n7. 📊 Admission grade\n8. 📈 Grade trend across semesters (engineered)\n9. 🎂 Age at enrollment\n10. 📜 Previous qualification grade""",
        "info_note_title": "📝 Important Notes",
        "info_note_body": """- 🌍 Model trained using student data from an international institution (Portugal)\n- 📊 GPA inputs are proportionally converted to dataset scale\n- 💹 Economic conditions use median dataset values\n- ⚠️ Predictions are indicative, not a final academic decision""",
        "info_built": "👨‍💻 Built by",
        "info_stack": "🛠️ Tech Stack",
        "info_star_note": "⭐ = Main model used",
        "error_sks1": "❌ Approved credits cannot exceed enrolled credits (Semester 1)",
        "error_sks2": "❌ Approved credits cannot exceed enrolled credits (Semester 2)",
        "proba_title": "📊 Prediction Probability",
        "proba_sub": "Neural Network",
        "labels": ["Dropout", "Enrolled", "Graduate"],
        "placeholder": "📝 Fill in student data in the form, then click 🚀 Predict Now",
        "processing": "⏳ Processing...",
    }
}

# ============================================================
# SESSION STATE
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "ID"
if "page" not in st.session_state:
    st.session_state.page = "predict"
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(_MODEL_DIR, 'scaler.pkl'))
    lr_model = joblib.load(os.path.join(_MODEL_DIR, 'logistic_model.pkl'))
    rf_model = joblib.load(os.path.join(_MODEL_DIR, 'random_forest_model.pkl'))
    nn_model = keras.models.load_model(os.path.join(_MODEL_DIR, 'neural_network_model.keras'))
    return scaler, lr_model, rf_model, nn_model

scaler, lr_model, rf_model, nn_model = load_models()

# ============================================================
# HELPER
# ============================================================
def ipk_to_grade(ipk):
    return 95 + (ipk / 4.0) * 95

def ipk_to_sem_grade(ipk):
    return (ipk / 4.0) * 18.57

# ============================================================
# CUSTOM CSS — CYBERPUNK THEME
# Cyan: #00f5ff  Magenta: #ff00aa  Purple: #7b2fff  Dark: #050815
# ============================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
    /* ── Root & Reset ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --cyan:    #00f5ff;
        --magenta: #ff00aa;
        --purple:  #7b2fff;
        --dark:    #050815;
        --dark2:   #0a1128;
        --dark3:   #0f1a3a;
        --card-bg: rgba(10, 17, 40, 0.85);
        --border:  rgba(0, 245, 255, 0.18);
        --text:    #c8d8f0;
        --text-dim:#6a82a8;
    }

    html, body, [class*="css"] {
        font-family: 'Rajdhani', 'Share Tech Mono', sans-serif;
        color: var(--text);
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* ── Streamlit label / text color fix ── */
    label, .stSlider label, .stNumberInput label,
    .stRadio label > div > p, .stSelectbox label, p, li, span {
        color: var(--text) !important;
    }
    .stRadio > div > label > div > p { color: var(--text) !important; }
    [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }

    /* ── Background ── */
    .stApp {
        background: var(--dark);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(123,47,255,0.25) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 90% 90%, rgba(0,245,255,0.12) 0%, transparent 60%),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0,245,255,0.015) 2px,
                rgba(0,245,255,0.015) 4px
            );
    }

    /* ── TOP HEADER BANNER ── */
    .gdgoc-header {
        background: linear-gradient(135deg, #0a1128 0%, #0d1f4a 50%, #120d2e 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 0 30px rgba(0,245,255,0.12), 0 0 60px rgba(123,47,255,0.08), inset 0 1px 0 rgba(0,245,255,0.1);
        position: relative;
        overflow: hidden;
    }
    .gdgoc-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), transparent);
        animation: scan-line 3s linear infinite;
    }
    @keyframes scan-line {
        0%   { opacity: 0.4; }
        50%  { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .gdgoc-header::after {
        content: '';
        position: absolute;
        right: -60px; top: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(123,47,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .gdgoc-dots {
        display: flex; gap: 8px; align-items: center;
    }
    .dot-red   { width:10px; height:10px; border-radius:50%; background:#ff0055; box-shadow:0 0 8px #ff0055; animation: pulse-dot 2s ease-in-out infinite; }
    .dot-yellow{ width:10px; height:10px; border-radius:50%; background:#ffcc00; box-shadow:0 0 8px #ffcc00; animation: pulse-dot 2s ease-in-out 0.3s infinite; }
    .dot-green { width:10px; height:10px; border-radius:50%; background:#00ff88; box-shadow:0 0 8px #00ff88; animation: pulse-dot 2s ease-in-out 0.6s infinite; }
    .dot-blue  { width:10px; height:10px; border-radius:50%; background:var(--cyan); box-shadow:0 0 8px var(--cyan); animation: pulse-dot 2s ease-in-out 0.9s infinite; }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.35; }
    }
    .header-title {
        color: var(--cyan);
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 0 0 20px rgba(0,245,255,0.6);
    }
    .header-badge {
        color: var(--cyan);
        font-family: 'Share Tech Mono', monospace;
        font-size: 11px;
        background: rgba(0,245,255,0.08);
        padding: 4px 14px;
        border-radius: 4px;
        border: 1px solid rgba(0,245,255,0.3);
        letter-spacing: 0.5px;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: center;
        border: 1px solid var(--border);
        box-shadow: 0 0 20px rgba(0,245,255,0.06);
        transition: transform 0.25s, box-shadow 0.25s;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .metric-card-blue::before   { background: linear-gradient(90deg, var(--cyan), var(--purple)); box-shadow: 0 0 10px var(--cyan); }
    .metric-card-red::before    { background: linear-gradient(90deg, #ff0055, #ff6600); box-shadow: 0 0 10px #ff0055; }
    .metric-card-yellow::before { background: linear-gradient(90deg, #ffcc00, var(--magenta)); box-shadow: 0 0 10px #ffcc00; }
    .metric-card-green::before  { background: linear-gradient(90deg, #00ff88, var(--cyan)); box-shadow: 0 0 10px #00ff88; }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(0,245,255,0.15), 0 8px 24px rgba(0,0,0,0.4);
        border-color: rgba(0,245,255,0.4);
    }
    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        color: var(--text-dim);
        margin: 0 0 6px;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .metric-value-blue   { color: var(--cyan);    text-shadow: 0 0 15px rgba(0,245,255,0.6); }
    .metric-value-red    { color: #ff4466;         text-shadow: 0 0 15px rgba(255,0,85,0.5); }
    .metric-value-yellow { color: #ffdd00;          text-shadow: 0 0 15px rgba(255,200,0,0.5); }
    .metric-value-green  { color: #00ff88;          text-shadow: 0 0 15px rgba(0,255,136,0.5); }
    .metric-sub {
        font-size: 10px;
        color: var(--text-dim);
        margin: 4px 0 0;
        font-family: 'Share Tech Mono', monospace;
    }

    /* ── SECTION CARDS / LABELS ── */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.3rem 1.4rem;
        height: 100%;
        box-shadow: 0 0 20px rgba(0,245,255,0.05);
    }
    .card-section-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.15em;
        color: var(--cyan);
        margin: 0 0 1rem;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        text-shadow: 0 0 10px rgba(0,245,255,0.5);
    }
    .card-sub-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 13px;
        color: var(--cyan);
        margin: 0 0 12px;
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 6px;
        background: rgba(0,245,255,0.05);
        border-left: 2px solid var(--cyan);
        letter-spacing: 0.5px;
    }
    .card-sub-label-finance {
        color: #00ff88;
        border-left-color: #00ff88;
        background: rgba(0,255,136,0.05);
    }
    .card-sub-label-academic {
        color: #ffcc00;
        border-left-color: #ffcc00;
        background: rgba(255,200,0,0.05);
    }

    /* ── RADIO BUTTON OVERRIDES ── */
    div[role="radiogroup"] {
        gap: 0px !important;
    }
    div[role="radiogroup"] label {
        background: rgba(10,17,40,0.6) !important;
        border: 1px solid rgba(0,245,255,0.2) !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        margin: 3px 0 !important;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
    }
    div[role="radiogroup"] label:hover {
        border-color: var(--cyan) !important;
        background: rgba(0,245,255,0.07) !important;
        box-shadow: 0 0 12px rgba(0,245,255,0.15) !important;
    }

    /* ── INPUT & SELECT OVERRIDES ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(10,17,40,0.8) !important;
        border: 1px solid rgba(0,245,255,0.25) !important;
        border-radius: 6px !important;
        color: var(--cyan) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 12px rgba(0,245,255,0.25) !important;
    }
    .stSelectbox > div > div {
        background: rgba(10,17,40,0.8) !important;
        border: 1px solid rgba(0,245,255,0.25) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
    }
    [data-baseweb="select"] > div {
        background: rgba(10,17,40,0.9) !important;
        border-color: rgba(0,245,255,0.25) !important;
    }

    /* ── SLIDER ── */
    .stSlider > div > div > div > div {
        background: var(--cyan) !important;
        box-shadow: 0 0 8px var(--cyan) !important;
    }

    /* ── RESULT CARDS ── */
    .result-card-success {
        background: linear-gradient(135deg, rgba(0,255,136,0.07) 0%, rgba(0,245,255,0.04) 100%);
        border: 1px solid rgba(0,255,136,0.35);
        border-left: 3px solid #00ff88;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 0 20px rgba(0,255,136,0.1), inset 0 1px 0 rgba(0,255,136,0.1);
    }
    .result-card-warning {
        background: linear-gradient(135deg, rgba(255,200,0,0.07) 0%, rgba(255,100,0,0.04) 100%);
        border: 1px solid rgba(255,200,0,0.35);
        border-left: 3px solid #ffcc00;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 0 20px rgba(255,200,0,0.1), inset 0 1px 0 rgba(255,200,0,0.1);
    }
    .result-card-danger {
        background: linear-gradient(135deg, rgba(255,0,85,0.07) 0%, rgba(255,0,170,0.04) 100%);
        border: 1px solid rgba(255,0,85,0.35);
        border-left: 3px solid #ff0055;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 0 20px rgba(255,0,85,0.1), inset 0 1px 0 rgba(255,0,85,0.1);
    }
    .result-pred-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 9px;
        font-weight: 400;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin: 0 0 6px;
        color: var(--text-dim);
    }
    .result-pred-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        font-weight: 800;
        margin: 0 0 6px;
        letter-spacing: 1px;
    }
    .result-pred-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        margin: 0;
        opacity: 0.6;
        letter-spacing: 0.3px;
    }

    /* ── RISK BAR ── */
    .risk-bar-wrap {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 10px 0;
        box-shadow: 0 0 15px rgba(0,245,255,0.05);
    }
    .risk-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ── MODEL COMPARISON ── */
    .model-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(10,17,40,0.6);
        border: 1px solid rgba(0,245,255,0.1);
        font-family: 'Rajdhani', sans-serif;
        font-size: 14px;
        transition: all 0.2s;
        color: var(--text);
    }
    .model-row:hover {
        background: rgba(0,245,255,0.05);
        border-color: rgba(0,245,255,0.3);
        box-shadow: 0 0 15px rgba(0,245,255,0.08);
    }
    .model-row-active {
        background: rgba(0,245,255,0.06);
        border: 1px solid rgba(0,245,255,0.3);
        box-shadow: 0 0 20px rgba(0,245,255,0.1);
    }
    .badge-pill {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-success { background: rgba(0,255,136,0.1);  color: #00ff88;  border: 1px solid rgba(0,255,136,0.25); }
    .badge-warning { background: rgba(255,200,0,0.1);   color: #ffcc00;  border: 1px solid rgba(255,200,0,0.25); }
    .badge-danger  { background: rgba(255,0,85,0.1);    color: #ff4466;  border: 1px solid rgba(255,0,85,0.25); }
    .badge-blue    { background: rgba(0,245,255,0.1);   color: var(--cyan); border: 1px solid rgba(0,245,255,0.25); }

    /* ── INFO PAGE ── */
    .info-section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 10px;
        font-weight: 700;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 1.3rem 0 0.5rem;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        background: rgba(0,245,255,0.05);
        border-left: 2px solid var(--cyan);
        text-shadow: 0 0 10px rgba(0,245,255,0.4);
    }
    .model-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-family: 'Rajdhani', sans-serif;
        border-radius: 8px;
        overflow: hidden;
    }
    .model-table th {
        text-align: left;
        padding: 8px 12px;
        background: rgba(0,245,255,0.1);
        color: var(--cyan);
        font-weight: 700;
        font-size: 11px;
        border-bottom: 1px solid rgba(0,245,255,0.2);
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.05em;
    }
    .model-table td {
        padding: 9px 12px;
        border-bottom: 1px solid rgba(0,245,255,0.07);
        color: var(--text);
    }
    .model-table tr:last-child td { border-bottom: none; }
    .model-table tr:nth-child(even) td { background: rgba(0,245,255,0.03); }
    .model-table tr:hover td { background: rgba(0,245,255,0.06); }

    /* ── SEPARATOR ── */
    .gdgoc-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), var(--magenta), transparent);
        margin: 0.8rem 0 1.2rem;
        border: none;
        box-shadow: 0 0 10px rgba(0,245,255,0.3);
    }

    /* ── FOOTER ── */
    .gdgoc-footer {
        text-align: center;
        padding: 1rem;
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        color: var(--text-dim);
        border-top: 1px solid rgba(0,245,255,0.1);
        margin-top: 1.5rem;
        letter-spacing: 0.3px;
    }
    .footer-dots {
        display: inline-flex; gap: 6px; margin-right: 10px; vertical-align: middle;
    }
    .footer-dot {
        width: 6px; height: 6px; border-radius: 50%; display: inline-block;
    }

    /* ── PRIMARY BUTTON ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0,245,255,0.15) 0%, rgba(123,47,255,0.2) 100%) !important;
        border: 1px solid var(--cyan) !important;
        border-radius: 6px !important;
        color: var(--cyan) !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        padding: 0.65rem !important;
        box-shadow: 0 0 20px rgba(0,245,255,0.2), inset 0 1px 0 rgba(0,245,255,0.1) !important;
        transition: all 0.25s !important;
        text-transform: uppercase !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0,245,255,0.25) 0%, rgba(123,47,255,0.3) 100%) !important;
        box-shadow: 0 0 35px rgba(0,245,255,0.4), 0 0 60px rgba(123,47,255,0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        background: rgba(0,245,255,0.04) !important;
        border: 1px solid rgba(0,245,255,0.15) !important;
        border-radius: 6px !important;
        color: var(--cyan) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] > details {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* ── ALERTS ── */
    .stAlert {
        background: rgba(10,17,40,0.8) !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] p { color: var(--text) !important; }

    /* ── RESPONSIVE ── */
    @media (max-width: 768px) {
        .header-title { font-size: 14px; }
        .block-container { padding-top: 0.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AMBIL TEKS SESUAI BAHASA
# ============================================================
L = LANG[st.session_state.lang]

# ============================================================
# TOP HEADER BANNER
# ============================================================
st.markdown(f"""
<div class="gdgoc-header">
    <div style="display:flex; align-items:center; gap:14px; z-index:1;">
        <div class="gdgoc-dots">
            <div class="dot-red"></div>
            <div class="dot-yellow"></div>
            <div class="dot-green"></div>
            <div class="dot-blue"></div>
        </div>
        <div>
            <p class="header-title">◈ {L['title']}</p>
            <p style="color:rgba(0,245,255,0.5); font-family:'Share Tech Mono',monospace; font-size:11px; margin:4px 0 0; letter-spacing:0.5px;">
                {L['subtitle']}
            </p>
        </div>
    </div>
    <div style="z-index:1;">
        <span class="header-badge">{L['badge']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAV (Language + Page switcher)
# ============================================================
col_nav2, col_nav3 = st.columns([4, 1])

with col_nav2:
    lang_options = ["🇮🇩 Indonesia", "🇬🇧 English"]
    current_lang_idx = 0 if st.session_state.lang == "ID" else 1
    selected_lang = st.selectbox(
        "Bahasa / Language",
        lang_options,
        index=current_lang_idx,
        label_visibility="collapsed"
    )
    new_lang = "ID" if selected_lang == "🇮🇩 Indonesia" else "EN"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

with col_nav3:
    page_options = [L['nav_predict'], L['nav_model']]
    current_page_idx = 0 if st.session_state.page == "predict" else 1
    selected_page = st.selectbox(
        "Page",
        page_options,
        index=current_page_idx,
        label_visibility="collapsed"
    )
    new_page = "predict" if selected_page == L['nav_predict'] else "model"
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.rerun()

# GDGoC Colorful Divider
st.markdown('<hr class="gdgoc-divider">', unsafe_allow_html=True)

# ============================================================
# METRIC CARDS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card metric-card-blue">
        <p class="metric-label">{L['metric_data']}</p>
        <p class="metric-value metric-value-blue">4.424</p>
        <p class="metric-sub">{L['metric_data_sub']}</p>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card metric-card-red">
        <p class="metric-label">{L['metric_acc']}</p>
        <p class="metric-value metric-value-red">78.08%</p>
        <p class="metric-sub">Random Forest</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card metric-card-yellow">
        <p class="metric-label">{L['metric_f1']}</p>
        <p class="metric-value metric-value-yellow">70.57%</p>
        <p class="metric-sub">Random Forest</p>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card metric-card-green">
        <p class="metric-label">{L['metric_models']}</p>
        <p class="metric-value metric-value-green">3</p>
        <p class="metric-sub">LR · RF · NN</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.2rem'></div>", unsafe_allow_html=True)

# ============================================================
# PAGE: PREDIKSI
# ============================================================
if st.session_state.page == "predict":

    col_input, col_result = st.columns([1, 1.4], gap="medium")

    with col_input:
        st.markdown(f'<div class="card-section-label">{L["section_input"]}</div>',
                    unsafe_allow_html=True)

        # ── Personal ──
        st.markdown(f'<p class="card-sub-label">{L["section_personal"]}</p>',
                    unsafe_allow_html=True)

        age = st.number_input(L['age_label'], 17, 60, 19)

        # Gender — radio horizontal
        gender_radio = st.radio(
            L['gender_label'],
            L['gender_opts'],
            index=0,
            horizontal=True,
            key="gender_radio"
        )
        gender_val_str = L['gender_vals'][L['gender_opts'].index(gender_radio)]

        # Displaced — radio horizontal
        displaced_radio = st.radio(
            L['displaced_label'],
            L['yes_no'],
            index=0,
            horizontal=True,
            key="displaced_radio"
        )
        displaced_val_str = L['yes_no_vals'][L['yes_no'].index(displaced_radio)]

        # ── Finansial ──
        st.markdown(f'<p class="card-sub-label card-sub-label-finance">{L["section_finance"]}</p>',
                    unsafe_allow_html=True)

        # Scholarship — radio horizontal
        scholarship_radio = st.radio(
            L['scholarship_label'],
            L['yes_no'],
            index=0,
            horizontal=True,
            key="scholarship_radio"
        )
        scholarship_val_str = L['yes_no_vals'][L['yes_no'].index(scholarship_radio)]

        # Debtor — radio horizontal
        debtor_radio = st.radio(
            L['debtor_label'],
            L['yes_no'],
            index=0,
            horizontal=True,
            key="debtor_radio"
        )
        debtor_val_str = L['yes_no_vals'][L['yes_no'].index(debtor_radio)]

        # Tuition — radio horizontal (Ya/Tidak reversed)
        tuition_radio = st.radio(
            L['tuition_label'],
            L['yes_no_rev'],
            index=0,
            horizontal=True,
            key="tuition_radio"
        )
        tuition_val_str = L['yes_no_rev_vals'][L['yes_no_rev'].index(tuition_radio)]

        # ── Akademik ──
        st.markdown(f'<p class="card-sub-label card-sub-label-academic">{L["section_academic"]}</p>',
                    unsafe_allow_html=True)

        ipk_masuk = st.slider(L['ipk_masuk_label'], 0.00, 4.00, 3.00, 0.01)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{L['sem1_header']}**")
            cu1_enrolled = st.number_input(L['sks_ambil'] + " (S1)", 0, 26, 20, key="cu1e")
            cu1_approved = st.number_input(L['sks_lulus'] + " (S1)", 0, 26, 18, key="cu1a")
            ipk_s1 = st.slider(L['ipk_sem'] + " 1", 0.00, 4.00, 3.00, 0.01, key="s1")
        with c2:
            st.markdown(f"**{L['sem2_header']}**")
            cu2_enrolled = st.number_input(L['sks_ambil'] + " (S2)", 0, 23, 20, key="cu2e")
            cu2_approved = st.number_input(L['sks_lulus'] + " (S2)", 0, 20, 18, key="cu2a")
            ipk_s2 = st.slider(L['ipk_sem'] + " 2", 0.00, 4.00, 3.00, 0.01, key="s2")

        predict_clicked = st.button(
            f"🚀 {L['predict_btn']}",
            use_container_width=True,
            type="primary"
        )

    # ============================================================
    # KOLOM HASIL
    # ============================================================
    with col_result:
        st.markdown(f'<div class="card-section-label">{L["result_section"]}</div>',
                    unsafe_allow_html=True)

        if predict_clicked:
            # ── Validasi ──
            if cu1_approved > cu1_enrolled:
                st.error(L['error_sks1'])
                st.stop()
            if cu2_approved > cu2_enrolled:
                st.error(L['error_sks2'])
                st.stop()

            # ── Mapping nilai ──
            def yn_map(val_str):
                return 1 if val_str in ("Ya", "Yes") else 0

            gender_encoded = 1 if gender_val_str in (L['gender_vals'][1],) else 0

            input_data = {
                'Marital status': 1, 'Application mode': 17,
                'Application order': 1, 'Course': 9238,
                'Daytime/evening attendance': 1, 'Previous qualification': 1,
                'Previous qualification (grade)': ipk_to_grade(ipk_masuk),
                'Nacionality': 1, "Mother's qualification": 19,
                "Father's qualification": 19, "Mother's occupation": 5,
                "Father's occupation": 5,
                'Admission grade': ipk_to_grade(ipk_masuk),
                'Displaced': yn_map(displaced_val_str),
                'Educational special needs': 0,
                'Debtor': yn_map(debtor_val_str),
                'Tuition fees up to date': yn_map(tuition_val_str),
                'Gender': gender_encoded,
                'Scholarship holder': yn_map(scholarship_val_str),
                'Age at enrollment': age, 'International': 0,
                'Curricular units 1st sem (credited)': 0,
                'Curricular units 1st sem (enrolled)': cu1_enrolled,
                'Curricular units 1st sem (evaluations)': cu1_enrolled,
                'Curricular units 1st sem (approved)': cu1_approved,
                'Curricular units 1st sem (grade)': ipk_to_sem_grade(ipk_s1),
                'Curricular units 1st sem (without evaluations)': 0,
                'Curricular units 2nd sem (credited)': 0,
                'Curricular units 2nd sem (enrolled)': cu2_enrolled,
                'Curricular units 2nd sem (evaluations)': cu2_enrolled,
                'Curricular units 2nd sem (approved)': cu2_approved,
                'Curricular units 2nd sem (grade)': ipk_to_sem_grade(ipk_s2),
                'Curricular units 2nd sem (without evaluations)': 0,
                'Unemployment rate': 11.1,
                'Inflation rate': 1.4,
                'GDP': 0.32,
                'approval_rate_sem1': cu1_approved / max(cu1_enrolled, 1),
                'approval_rate_sem2': cu2_approved / max(cu2_enrolled, 1),
                'grade_trend': ipk_to_sem_grade(ipk_s2) - ipk_to_sem_grade(ipk_s1),
            }

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            # ── Prediksi ──
            lr_pred  = lr_model.predict(input_scaled)[0]
            rf_pred  = rf_model.predict(input_scaled)[0]
            nn_proba = nn_model.predict(input_scaled, verbose=0)[0]
            nn_pred  = np.argmax(nn_proba)

            labels    = L['labels']
            label_map = {0: labels[0], 1: labels[1], 2: labels[2]}

            card_class  = {0: "result-card-danger",  1: "result-card-warning",  2: "result-card-success"}
            text_color  = {0: "#c5221f",             1: "#F9AB00",              2: "#1e8e3e"}
            badge_class = {0: "badge-danger",         1: "badge-warning",        2: "badge-success"}
            emoji_pred  = {0: "🚨",                   1: "⚠️",                   2: "🎓"}

            rf_label = label_map[rf_pred]
            rf_class = card_class[rf_pred]
            rf_color = text_color[rf_pred]
            rf_emoji = emoji_pred[rf_pred]

            # ── Hasil Utama ──
            st.markdown(f"""
            <div class="{rf_class}">
                <p class="result-pred-label" style="color:{rf_color};">
                    {L['main_pred_label']}
                </p>
                <p class="result-pred-value" style="color:{rf_color};">
                    {rf_emoji} {rf_label}
                </p>
                <p class="result-pred-sub" style="color:{rf_color};">
                    {L['based_on']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

            # ── Risk Bar ──
            risk_score = nn_proba[0] * 100
            if risk_score < 30:
                risk_color = "#34A853"
                risk_msg   = L['risk_low']
            elif risk_score < 60:
                risk_color = "#FBBC05"
                risk_msg   = L['risk_mid']
            else:
                risk_color = "#EA4335"
                risk_msg   = L['risk_high']

            st.markdown(f"""
            <div class="risk-bar-wrap">
                <p class="card-section-label" style="margin-bottom:4px;">
                    {L['risk_title']}
                </p>
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                    <div class="risk-bar-container" style="flex:1;">
                        <div style="width:{risk_score:.1f}%; height:100%;
                             background:linear-gradient(90deg, {risk_color}aa, {risk_color});
                             border-radius:999px; transition:width 0.5s ease;"></div>
                    </div>
                    <span style="font-size:18px; font-weight:700; color:{risk_color}; min-width:48px;">
                        {risk_score:.1f}%
                    </span>
                </div>
                <p style="font-size:12px; color:{risk_color}; font-weight:600; margin:0;">
                    {risk_msg}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Probabilitas Chart ──
            with st.expander(f"{L['proba_title']} ({L['proba_sub']})", expanded=True):
                fig, ax = plt.subplots(figsize=(5, 2.4))
                cyber_colors = ["#ff0055", "#ffcc00", "#00ff88"]
                bars = ax.barh(labels, nn_proba, color=cyber_colors, height=0.45,
                               edgecolor='none', linewidth=0)
                ax.set_xlim(0, 1.15)
                ax.set_xlabel("Probability", fontsize=9, color='#6a82a8')
                ax.tick_params(labelsize=9, colors='#6a82a8')
                ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
                ax.set_facecolor('none')
                for bar, val, col in zip(bars, nn_proba, cyber_colors):
                    ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                            f"{val * 100:.1f}%", va='center',
                            fontsize=9, fontweight='bold', color=col)
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                ax.xaxis.label.set_color('#6a82a8')
                ax.tick_params(axis='y', colors='#c8d8f0')
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── Perbandingan Model ──
            st.markdown(f"""
            <div class="card-section-label" style="margin-top:12px;">
                {L['comparison_title']}
            </div>
            """, unsafe_allow_html=True)

            model_names = ["Logistic Regression", "Random Forest", "Neural Network"]
            model_icons = ["📉", "🌲", "🧠"]
            model_preds = [lr_pred, rf_pred, nn_pred]
            model_accs  = ["76.72%", "78.08%", "76.27%"]
            is_main     = [False, True, False]

            for name, icon, pred, acc, main in zip(
                    model_names, model_icons, model_preds, model_accs, is_main):
                row_class = "model-row model-row-active" if main else "model-row"
                main_tag  = (f'<span class="badge-pill badge-blue" '
                             f'style="font-size:10px; margin-left:6px;">'
                             f'⭐ {L["main_badge"]}</span>') if main else ""
                pred_bclass = badge_class[pred]
                st.markdown(f"""
                <div class="{row_class}">
                    <span>{icon} {name}{main_tag}</span>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span class="badge-pill {pred_bclass}">
                            {emoji_pred[pred]} {label_map[pred]}
                        </span>
                        <span style="font-size:11px; color:#9aa0a6;">
                            🎯 {acc}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Rekomendasi ──
            st.markdown("---")
            st.markdown(f"**{L['rec_title']}**")
            if rf_pred == 0:
                st.error(L['rec_dropout'])
            elif rf_pred == 1:
                st.warning(L['rec_enrolled'])
            else:
                st.success(L['rec_graduate'])

            # ── Disclaimer ──
            st.markdown(f"""
            <p style="font-family:'Share Tech Mono',monospace; font-size:10px; color:rgba(0,245,255,0.35); margin-top:12px; text-align:center; letter-spacing:0.5px;">
                {L['disclaimer']}
            </p>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:rgba(10,17,40,0.7); border:1px dashed rgba(0,245,255,0.25);
                        border-radius:10px; padding:3rem 2rem; text-align:center;">
                <div style="font-size:42px; margin-bottom:1rem; color:#00f5ff; filter:drop-shadow(0 0 12px #00f5ff);">◈</div>
                <p style="font-family:'Orbitron',sans-serif; font-size:10px; font-weight:600;
                           color:rgba(0,245,255,0.7); letter-spacing:1.5px; margin:0 0 10px; text-transform:uppercase;">
                    AWAITING INPUT
                </p>
                <p style="font-family:'Share Tech Mono',monospace; font-size:12px; color:rgba(0,245,255,0.45); margin:0 0 8px;">
                    {L['placeholder']}
                </p>
                <p style="font-family:'Share Tech Mono',monospace; font-size:11px; color:rgba(0,245,255,0.3); margin:0; letter-spacing:0.5px;">
                    [ LR ] &nbsp;·&nbsp; [ RF ] &nbsp;·&nbsp; [ NN ]
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: INFO MODEL
# ============================================================
elif st.session_state.page == "model":

    st.markdown(f"## {L['info_title']}")

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown(f'<p class="info-section-title">{L["info_dataset_title"]}</p>',
                    unsafe_allow_html=True)
        st.markdown(L['info_dataset_body'])

        st.markdown(f'<p class="info-section-title">{L["info_feature_title"]}</p>',
                    unsafe_allow_html=True)
        st.markdown(L['info_feature_body'])

        st.markdown(f'<p class="info-section-title">{L["info_note_title"]}</p>',
                    unsafe_allow_html=True)
        st.markdown(L['info_note_body'])

    with col_b:
        st.markdown(f'<p class="info-section-title">{L["info_model_title"]}</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <table class="model-table">
            <tr>
                <th>🤖 Model</th>
                <th>🎯 Accuracy</th>
                <th>📊 F1 Macro</th>
                <th>⚖️ F1 Weighted</th>
            </tr>
            <tr>
                <td>📉 Logistic Regression</td>
                <td>76.72%</td><td>67.71%</td><td>75.06%</td>
            </tr>
            <tr style="background:rgba(66,133,244,0.06);">
                <td><strong>🌲 Random Forest ⭐</strong></td>
                <td>78.08%</td><td><strong>70.57%</strong></td><td>76.79%</td>
            </tr>
            <tr>
                <td>🧠 Neural Network</td>
                <td>76.27%</td><td>69.80%</td><td>75.85%</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-size:11px; color:#9aa0a6; margin-top:-4px;">{L["info_star_note"]}</p>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance plot
        try:
            img = plt.imread(os.path.join(_DATA_DIR, 'feature_importance.png'))
            st.image(img, use_container_width=True)
        except Exception:
            st.caption("📊 Feature importance chart tidak tersedia.")

        st.markdown(f'<p class="info-section-title">{L["info_built"]}</p>',
                    unsafe_allow_html=True)
        st.markdown("🏫 Hannan Rava Mahardika - Member GDGoC USU")

        st.markdown(f'<p class="info-section-title">{L["info_stack"]}</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "🐍 Python &nbsp;·&nbsp; 🔬 Scikit-learn &nbsp;·&nbsp; "
            "🧠 TensorFlow/Keras &nbsp;·&nbsp; 🌊 Streamlit &nbsp;·&nbsp; "
            "🐼 Pandas &nbsp;·&nbsp; 🔢 NumPy &nbsp;·&nbsp; 📊 Matplotlib"
        )

    # ── Footer ──
    st.markdown(f"""
    <div class="gdgoc-footer">
        <span class="footer-dots">
            <span class="footer-dot" style="background:#00f5ff; box-shadow:0 0 6px #00f5ff;"></span>
            <span class="footer-dot" style="background:#7b2fff; box-shadow:0 0 6px #7b2fff;"></span>
            <span class="footer-dot" style="background:#ff00aa; box-shadow:0 0 6px #ff00aa;"></span>
            <span class="footer-dot" style="background:#00ff88; box-shadow:0 0 6px #00ff88;"></span>
        </span>
        {L['disclaimer']}
    </div>
    """, unsafe_allow_html=True)