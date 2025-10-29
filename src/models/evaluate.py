import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, KFold
from sklearn.metrics import r2_score, mean_squared_error
from src.utils.io import load_model, ensure_dir, save_json
from src.data.generate import RAW_PATH
from src.visualization.plots import plot_pred_vs_actual, plot_residuals, plot_learning_curve, plot_model_comparison
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

BEST_MODEL_PATH = "models/best_model.joblib"

def _load_data():
    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y

def _build_models_for_comparison():
    models = []
    for deg in [1, 2, 3, 4, 5]:
        # Linear
        models.append(("LR_d"+str(deg), Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", LinearRegression())
        ])))
        # Ridge
        models.append(("Ridge_d"+str(deg), Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0))
        ])))
        # Lasso
        models.append(("Lasso_d"+str(deg), Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Lasso(alpha=0.1, max_iter=10000))
        ])))
    return models

def evaluate_and_plot(random_state=42):
    X, y = _load_data()
    model = load_model(BEST_MODEL_PATH)

    # Pred vs actual & residuals (full data predict, hanya visual analisis)
    yhat = model.predict(X)
    plot_pred_vs_actual(y, yhat, out_path="reports/figures/pred_vs_actual.png")
    plot_residuals(y, yhat, out_path="reports/figures/residuals.png")

    # Learning curve (R^2 vs m)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    plot_learning_curve(model, X, y, cv=cv, out_path="reports/figures/learning_curve.png")

    # Model comparison (R^2 via CV pada full data untuk ringkasan)
    models = _build_models_for_comparison()
    names, scores = [], []
    for name, pipe in models:
        cv_scores = []
        for tr_idx, te_idx in cv.split(X):
            pipe.fit(X[tr_idx], y[tr_idx])
            y_pred = pipe.predict(X[te_idx])
            cv_scores.append(r2_score(y[te_idx], y_pred))
        names.append(name)
        scores.append(np.mean(cv_scores))

    plot_model_comparison(names, scores, out_path="reports/figures/model_comparison_r2.png")
    save_json({"labels": names, "mean_cv_r2": scores}, "reports/model_comparison_summary.json")
    print("Evaluasi & plotting selesai. PNG tersimpan di reports/figures/.")