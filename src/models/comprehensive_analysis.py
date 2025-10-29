import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils.io import ensure_dir, save_model, save_json, save_csv
from src.visualization.plots import (
    plot_pred_vs_actual_detailed, 
    plot_residuals_detailed, 
    plot_learning_curve_detailed,
    plot_coefficient_importance
)
from src.visualization.exploratory import create_all_exploratory_plots

RAW_PATH = "data/raw/synthetic_regression.csv"
OUTPUT_DIR = "reports/figures"

def _load_data():
    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y, df

def _get_model_configs():
    """Konfigurasi model yang akan dianalisis"""
    configs = []
    
    # Linear Regression dengan degree 1
    configs.append({
        'name': 'Linear',
        'degree': 1,
        'model': Pipeline([
            ("poly", PolynomialFeatures(degree=1, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", LinearRegression())
        ])
    })
    
    # Lasso dengan berbagai degree
    for degree in [2, 3, 4, 5]:
        configs.append({
            'name': 'Lasso',
            'degree': degree,
            'model': Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler(with_mean=False)),
                ("reg", Lasso(alpha=0.1, max_iter=10000))
            ])
        })
    
    return configs

def comprehensive_analysis(random_state=42, test_size=0.2):
    """Melakukan analisis komprehensif dan menghasilkan semua plot"""
    print("Memulai analisis komprehensif...")
    
    # Load data
    X, y, df = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Buat direktori output
    ensure_dir(OUTPUT_DIR)
    
    # 1. Buat plot eksploratori
    print("Membuat plot eksploratori...")
    create_all_exploratory_plots(RAW_PATH, OUTPUT_DIR)
    
    # 2. Analisis untuk setiap model
    model_configs = _get_model_configs()
    
    for config in model_configs:
        model_name = config['name']
        degree = config['degree']
        model = config['model']
        
        print(f"Menganalisis {model_name} degree {degree}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Prediksi
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Plot prediction vs actual
        plot_pred_vs_actual_detailed(y_train, y_train_pred, model_name, degree, "train", OUTPUT_DIR)
        plot_pred_vs_actual_detailed(y_test, y_test_pred, model_name, degree, "test", OUTPUT_DIR)
        
        # Plot residuals
        plot_residuals_detailed(y_train, y_train_pred, model_name, degree, "train", OUTPUT_DIR)
        plot_residuals_detailed(y_test, y_test_pred, model_name, degree, "test", OUTPUT_DIR)
        
        # Plot learning curve
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        plot_learning_curve_detailed(model, X_train, y_train, model_name, degree, cv, OUTPUT_DIR)
        
        # Plot coefficient importance (hanya untuk model yang memiliki coef_)
        try:
            feature_names = [f"x{i+1}" for i in range(X.shape[1])]
            plot_coefficient_importance(model, feature_names, model_name, degree, OUTPUT_DIR)
        except Exception as e:
            print(f"Tidak bisa membuat coefficient plot untuk {model_name} degree {degree}: {e}")
    
    print("Analisis komprehensif selesai!")
    print(f"Semua plot disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    comprehensive_analysis()