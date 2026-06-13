# 🎓 Prediksi Risiko Dropout Mahasiswa

> Machine Learning project untuk memprediksi risiko dropout mahasiswa berdasarkan data akademik dan demografis — dibangun sebagai Final Project **Machine Learning Pathway GDGoC USU**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)

## 🌐 Live Demo
👉 [Coba Aplikasi Sekarang](https://student-dropout-prediction-dbfmkgannk9qwfzrnypfta.streamlit.app/)

---

## 📌 Deskripsi Proyek

Dropout mahasiswa adalah salah satu tantangan terbesar institusi pendidikan tinggi. Proyek ini membangun sistem prediksi berbasis Machine Learning yang mampu mengidentifikasi mahasiswa berisiko tinggi dropout sejak dini, sehingga pihak akademik dapat mengambil tindakan preventif lebih awal.

Model dilatih menggunakan dataset **Student Dropout and Academic Success** dari UCI ML Repository (4.424 data mahasiswa, 36 fitur original + 3 fitur engineered) dengan tiga algoritma berbeda yang dibandingkan performanya secara langsung.

---

## 🚀 Demo Aplikasi

Dua aplikasi tersedia dalam repositori ini:

### 🌊 Streamlit App (Utama)

Aplikasi prediksi interaktif dengan tema cyberpunk, mendukung dua bahasa.

**Fitur utama:**
- 🔍 Prediksi risiko dropout berdasarkan data akademik mahasiswa
- 📊 Visualisasi probabilitas prediksi dari Neural Network
- 🤖 Perbandingan hasil dari 3 model ML sekaligus
- 🌡️ Risk score meter dengan indikator warna (rendah/sedang/tinggi)
- 💡 Rekomendasi tindakan berdasarkan hasil prediksi
- 🌐 Dukungan dua bahasa: **Bahasa Indonesia** dan **English**
- 🎨 Tema cyberpunk dengan aksen cyan, magenta, dan purple

### ⚛️ Web Frontend (Pendukung)

Frontend berbasis **Vite + React + TypeScript + Tailwind CSS** yang dibuild menjadi _single-file HTML_ (tersedia di `dist/index.html`). Dapat dibuka langsung di browser tanpa server.

---

## 📊 Hasil Model

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| Logistic Regression | 76.72% | 67.71% | 75.06% |
| **Random Forest ⭐** | **78.08%** | **70.57%** | **76.79%** |
| Neural Network | 76.27% | 69.80% | 75.85% |

> ⭐ **Random Forest** dipilih sebagai model utama karena memiliki F1 Macro tertinggi (70.57%) — metrik paling relevan untuk data dengan 3 kelas tidak seimbang.

### Detail Per-Kelas (Random Forest)

| Kelas | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Dropout | 0.81 | 0.77 | 0.79 |
| Enrolled | 0.59 | 0.38 | 0.46 |
| Graduate | 0.81 | 0.93 | 0.86 |

---

## 🗂️ Struktur Proyek

```
dropout-prediction/
├── app/
│   └── app.py                    # Streamlit web application (1200 baris)
├── data/
│   ├── dataset.csv               # Dataset utama (UCI ML Repository)
│   ├── confusion_matrices.png    # Confusion matrix ketiga model
│   ├── eda_plots.png             # Visualisasi EDA
│   ├── feature_importance.png    # Feature importance Random Forest
│   ├── heatmap.png               # Heatmap korelasi fitur
│   ├── learning_curve.png        # Learning curve Neural Network
│   └── plot_target.png           # Distribusi target
├── dist/
│   └── index.html                # Frontend build output (single-file)
├── model/
│   ├── logistic_model.pkl        # Trained Logistic Regression
│   ├── random_forest_model.pkl   # Trained Random Forest
│   ├── neural_network_model.keras # Trained Neural Network (15.171 params)
│   └── scaler.pkl                # StandardScaler (fitted on training data)
├── notebook/
│   └── analysis.ipynb            # EDA + Preprocessing + Training + Evaluasi
├── src/                          # Frontend source (Vite + React + TS)
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
├── index.html                    # Frontend entry point
├── package.json                  # Node.js dependencies
├── tsconfig.json                 # TypeScript configuration
├── vite.config.ts                # Vite bundler configuration
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning & Backend

| Kategori | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualisasi | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Deployment | Streamlit |
| Model Saving | Joblib, Keras (.keras) |
| Environment | Jupyter Notebook / VS Code |

### Frontend

| Kategori | Tools |
|----------|-------|
| Language | TypeScript 5.9 |
| Framework | React 19, Vite 7 |
| Styling | Tailwind CSS 4 |
| Bundling | vite-plugin-singlefile |

---

## ⚙️ Instalasi & Cara Menjalankan

### Prasyarat

- Python 3.10+
- Node.js 18+ (untuk frontend)
- pip dan venv

### 1. Clone repositori

```bash
git clone https://github.com/username/dropout-prediction.git
cd dropout-prediction
```

### 2. Setup Backend (Streamlit App)

```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit joblib

# Jalankan Streamlit app
streamlit run app/app.py
```

Buka browser dan akses: **http://localhost:8501**

### 3. Setup Frontend (Opsional)

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build ke single-file HTML
npm run build
```

Hasil build tersedia di `dist/index.html` — dapat dibuka langsung di browser.

---

## 📓 Menjalankan Notebook

Untuk melihat proses lengkap EDA, preprocessing, training, dan evaluasi:

```bash
jupyter notebook notebook/analysis.ipynb
```

Atau gunakan VS Code untuk membuka file `.ipynb` langsung.

---

## 🔄 ML Pipeline

```
Dataset (UCI)
│
▼
Exploratory Data Analysis (EDA)
│   - Distribusi target
│   - Heatmap korelasi
│   - Analisis dropout per kategori
│   - Distribusi usia
│
▼
Data Preprocessing
│   - Strip whitespace dari nama kolom
│   - Label encoding target (Dropout→0, Enrolled→1, Graduate→2)
│   - Train-test split (80/20, stratified, random_state=42)
│   - StandardScaler (fit hanya di train, transform train + test)
│
▼
Feature Engineering
│   - approval_rate_sem1 = approved_1 / max(enrolled_1, 1)
│   - approval_rate_sem2 = approved_2 / max(enrolled_2, 1)
│   - grade_trend = grade_sem2 - grade_sem1
│
▼
Model Training
│   - Logistic Regression (default params)
│   - Random Forest (100 estimators, random_state=42, n_jobs=-1)
│   - Neural Network (128→64→32→3, Dropout 0.3 & 0.2, Adam, 50 epochs)
│
▼
Model Evaluation
│   - Classification report (precision, recall, f1 per kelas)
│   - Confusion matrix
│   - Learning curve
│   - Perbandingan F1 Macro
│
▼
Deployment (Streamlit)
│   - Load model dari file (.pkl / .keras)
│   - Input konversi IPK (0–4) → skala dataset
│   - Prediksi real-time
│   - Visualisasi probabilitas + risk meter
└   - Rekomendasi tindakan
```

---

## 📁 Dataset

- **Nama:** Student Dropout and Academic Success
- **Sumber:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Jumlah data:** 4.424 mahasiswa
- **Jumlah fitur:** 36 original + 3 engineered = **39 total** (akademik, demografis, ekonomi)
- **Target:** Dropout (0) / Enrolled (1) / Graduate (2)

### Top 10 Fitur Paling Berpengaruh (Random Forest)

| Rank | Fitur | Importance |
|------|-------|------------|
| 1 | approval_rate_sem2 ⭐ | 0.1235 |
| 2 | Curricular units 2nd sem (approved) | 0.0900 |
| 3 | approval_rate_sem1 ⭐ | 0.0737 |
| 4 | Curricular units 2nd sem (grade) | 0.0636 |
| 5 | Curricular units 1st sem (approved) | 0.0558 |
| 6 | Curricular units 1st sem (grade) | 0.0521 |
| 7 | Admission grade | 0.0383 |
| 8 | grade_trend ⭐ | 0.0364 |
| 9 | Age at enrollment | 0.0347 |
| 10 | Previous qualification (grade) | 0.0340 |

---

## 🧠 Neural Network Architecture

```
Input (39)
    ↓
Dense (128, ReLU)
    ↓ Dropout (0.3)
Dense (64, ReLU)
    ↓ Dropout (0.2)
Dense (32, ReLU)
    ↓
Dense (3, Softmax)
```

- **Total params:** 15.555 (semua trainable)
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 50 | **Batch size:** 32 | **Validation split:** 0.2

---

## ⚠️ Catatan Penting

- Model dilatih menggunakan data mahasiswa dari institusi internasional (**Portugal**, UCI ML Repository)
- Input IPK (skala 0–4) dikonversi secara proporsional ke skala dataset asli
- Kondisi ekonomi (GDP, inflasi, unemployment) menggunakan nilai median dataset
- Prediksi bersifat **indikatif** sebagai alat bantu — bukan keputusan akademik final
- Performa model pada kelas "Enrolled" masih rendah (F1=0.43) — perlu improvement pada identifikasi mahasiswa aktif berisiko

---

## 👤 Author

**Hannan Rava Mahardika**
Machine Learning Pathway — GDGoC USU (Google Developer Groups on Campus, Universitas Sumatera Utara)

---

## 🏫 Tentang GDGoC USU

Proyek ini dibuat sebagai **Final Project Machine Learning Pathway** dari program **Google Developer Groups on Campus Universitas Sumatera Utara (GDGoC USU)**.

---

<p align="center">
  <sub>Dibuat dengan ❤️ untuk GDGoC USU Machine Learning Pathway</sub>
</p>
