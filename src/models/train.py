import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils.io import ensure_dir, save_model, save_json, save_csv

RAW_PATH = "data/raw/synthetic_regression.csv"
BEST_MODEL_PATH = "models/best_model.joblib"
METRICS_ALL_PATH = "reports/metrics_all.csv"
METRICS_BEST_JSON = "reports/metrics_best.json"

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)
    r2 = r2_score(y_true, y_pred)
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}

def _load_data():
    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y

def _candidate_models():
    base = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse safety
        ("reg", LinearRegression())
    ])

    # Grid for LinearRegression: only degree
    grid_lr = {
        "poly__degree": [1, 2, 3, 4, 5],
        "reg": [LinearRegression()]
    }

    # Ridge
    grid_ridge = {
        "poly__degree": [1, 2, 3, 4, 5],
        "reg": [Ridge()],
        "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    # Lasso
    grid_lasso = {
        "poly__degree": [1, 2, 3, 4, 5],
        "reg": [Lasso(max_iter=10000)],
        "reg__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
    }

    return base, [grid_lr, grid_ridge, grid_lasso]

def train_and_select(random_state=42, test_size=0.2):
    X, y = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base, grids = _candidate_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=base,
        param_grid=grids,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_tr_pred = best_model.predict(X_train)
    y_te_pred = best_model.predict(X_test)

    m_tr = _metrics(y_train, y_tr_pred)
    m_te = _metrics(y_test, y_te_pred)

    # Collect all results (top combos)
    rows = []
    header = ["rank", "mean_cv_r2", "std_cv_r2", "params"]
    for rank, (mean, std, params) in enumerate(
        zip(search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            search.cv_results_["params"]), start=1
    ):
        rows.append([rank, mean, std, str(params)])

    save_csv(rows, header, METRICS_ALL_PATH)
    
    # Konversi best_params agar bisa di-serialize ke JSON
    serializable_params = {}
    for key, value in search.best_params_.items():
        # Cek apakah value adalah objek sklearn model
        if hasattr(value, 'fit') and hasattr(value, 'predict'):
            # Jika value adalah objek model, ambil nama class-nya
            serializable_params[key] = value.__class__.__name__
        else:
            serializable_params[key] = value
    
    save_json({"train": m_tr, "test": m_te, "best_params": serializable_params}, METRICS_BEST_JSON)
    save_model(best_model, BEST_MODEL_PATH)

    print("Model terbaik disimpan ke:", BEST_MODEL_PATH)
    print("Metrik (train):", m_tr)
    print("Metrik (test):", m_te)
    return BEST_MODEL_PATH, METRICS_BEST_JSON, METRICS_ALL_PATH