import numpy as np
import pandas as pd
from src.utils.io import ensure_dir
import os

RAW_PATH = "data/raw/synthetic_regression.csv"

def _true_function(X):
    # X shape: (n, 5) -> columns x1..x5
    x1, x2, x3, x4, x5 = X.T
    # Non-linear target to justify polynomial features
    y = (
        3.0 * x1
        - 2.5 * x2**2
        + 1.2 * x3**3
        + 0.8 * x1 * x2
        - 1.5 * x4 * x5
        + 2.0 * np.sin(x3)
        + 5.0
    )
    return y

def generate_synthetic(n_samples=400, noise=3.0, random_state=42):
    rng = np.random.default_rng(random_state)
    # 5 fitur dengan rentang berbeda agar scaling relevan
    x1 = rng.normal(0, 1.0, n_samples)
    x2 = rng.uniform(-2, 2, n_samples)
    x3 = rng.normal(1.0, 1.5, n_samples)
    x4 = rng.uniform(0, 3, n_samples)
    x5 = rng.normal(-1.0, 0.5, n_samples)

    X = np.column_stack([x1, x2, x3, x4, x5])
    y = _true_function(X) + rng.normal(0, noise, n_samples)

    cols = ["x1", "x2", "x3", "x4", "x5"]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df

def generate_and_save(n_samples=400, noise=3.0):
    df = generate_synthetic(n_samples, noise)
    ensure_dir(os.path.dirname(RAW_PATH))
    df.to_csv(RAW_PATH, index=False)
    print(f"Dataset sintetis tersimpan: {RAW_PATH} (n={len(df)})")