# UTS — Polynomial Regression (Sklearn)

Proyek Python murni (siap di VS Code) untuk tugas **Polynomial Regression** dengan **GridSearchCV** pada derajat 1–5, regularisasi **Ridge/Lasso**, **K-Fold CV (k=5)**, **learning curve**, dan visualisasi residual serta _predicted vs actual_. Menggunakan dataset sintetis (400 baris).

## Struktur Proyek
```
uts-polynomial-regression/
├─ data/
│  └─ raw/
│     └─ synthetic_regression.csv    # dataset sintetis (400 baris)
├─ models/
│  └─ best_model.joblib              # model terbaik (pipeline lengkap)
├─ reports/
│  ├─ figures/                       # 44 file gambar PNG
│  │  ├─ hist_*.png                  # histogram untuk setiap fitur (5 file)
│  │  ├─ scatter_*_vs_Harga.png      # scatter plots (5 file)
│  │  ├─ corr_heatmap.png            # correlation heatmap
│  │  ├─ pred_vs_true_*_*.png        # prediksi vs aktual (12 file)
│  │  ├─ residuals_*_*.png           # residual plots (12 file)
│  │  ├─ learning_curve_*.png        # learning curves (5 file)
│  │  ├─ coeff_importance_*.png      # coefficient importance (4 file)
│  │  └─ *.png                       # file legacy dan lainnya
│  ├─ metrics_all.csv                # metrik semua model
│  ├─ metrics_best.json              # metrik model terbaik
│  └─ model_comparison_summary.json  # ringkasan perbandingan model
├─ src/
│  ├─ data/
│  │  └─ generate.py                 # generator data sintetis
│  ├─ models/
│  │  ├─ train.py                    # training + model selection
│  │  ├─ evaluate.py                 # metrik, CV, learning curve
│  │  └─ comprehensive_analysis.py   # analisis komprehensif semua model
│  ├─ utils/
│  │  └─ io.py                       # helper I/O (save/load, write metrics)
│  └─ visualization/
│     ├─ plots.py                    # fungsi plotting utama
│     └─ exploratory.py              # plot eksploratori (histogram, scatter, heatmap)
├─ .vscode/
│  ├─ launch.json
│  └─ settings.json
├─ venv/                             # virtual environment
├─ main.py                           # entry point utama
├─ requirements.txt                  # dependencies
└─ .gitignore
```

## Cara Menjalankan (Langkah Cepat)

### 1. Setup Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Dataset (Opsional)
```bash
python main.py --mode generate --n 400 --noise 3.0
```

### 3. Training Model
```bash
python main.py --mode train
```

### 4. Evaluasi Model
```bash
python main.py --mode evaluate
```

### 5. Analisis Komprehensif (Menghasilkan Semua Plot)
```bash
python -m src.models.comprehensive_analysis
```

## Output yang Dihasilkan

### Model & Metrik
- **Model terbaik**: `models/best_model.joblib` (pipeline lengkap: PolynomialFeatures + StandardScaler + Regressor)
- **Metrik terbaik**: `reports/metrics_best.json`
- **Metrik semua model**: `reports/metrics_all.csv`
- **Ringkasan perbandingan**: `reports/model_comparison_summary.json`

### Visualisasi (44 File PNG di `reports/figures/`)

#### Plot Eksploratori
- **Histogram**: `hist_AksesTransport.png`, `hist_Harga.png`, `hist_JarakPusat.png`, `hist_Kamar.png`, `hist_KamarMandi.png`
- **Scatter Plots**: `scatter_*_vs_Harga.png` (5 file)
- **Correlation Heatmap**: `corr_heatmap.png`

#### Plot Model Performance
- **Prediksi vs Aktual**: `pred_vs_true_{model}_{degree}_{train/test}.png` (12 file)
- **Residual Plots**: `residuals_{model}_{degree}_{train/test}.png` (12 file)
- **Learning Curves**: `learning_curve_{model}_{degree}.png` (5 file)
- **Coefficient Importance**: `coeff_importance_{model}_{degree}.png` (4 file)

#### Plot Legacy
- `pred_vs_actual.png`, `residuals.png`, `learning_curve.png`, `model_comparison_r2.png`

## Model yang Dianalisis
- **Linear Regression** (degree 1)
- **Lasso Regression** (degree 2, 3, 4, 5)
- Setiap model menggunakan **PolynomialFeatures** + **StandardScaler**
- **GridSearchCV** dengan **5-Fold Cross Validation**

## Fitur Utama
✅ **Dataset sintetis** dengan 5 fitur (Kamar, KamarMandi, JarakPusat, AksesTransport, Harga)  
✅ **Polynomial regression** dengan degree 1-5  
✅ **Regularisasi** Ridge dan Lasso  
✅ **Cross validation** K-Fold (k=5)  
✅ **Learning curves** untuk setiap model  
✅ **Visualisasi lengkap** (44 plot)  
✅ **Model persistence** dengan joblib  
✅ **Analisis eksploratori** data  
✅ **Perbandingan model** komprehensif  

## Catatan Teknis
- Menggunakan **scikit-learn** (tanpa implementasi manual)
- Bahasa kode dan dokumentasi: **Indonesia**
- Semua grafik disimpan sebagai **PNG**
- Pipeline model sudah include preprocessing
- Auto-create direktori untuk output

## Cara Menggunakan Model Tersimpan
```python
import joblib
import numpy as np

# Load model (sudah include preprocessing)
model = joblib.load('models/best_model.joblib')

# Prediksi data baru
X_new = np.array([[3, 2, 5.5, 8, 0]])  # [Kamar, KamarMandi, JarakPusat, AksesTransport, dummy]
prediction = model.predict(X_new)
print(f"Prediksi harga: {prediction[0]:.2f}")
```
