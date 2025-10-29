import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.io import ensure_dir
from sklearn.model_selection import learning_curve

def _simple_save(fig, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_pred_vs_actual_detailed(y_true, y_pred, model_name, degree, dataset_type, out_dir="reports/figures"):
    """Plot prediction vs actual dengan nama file yang detail"""
    ensure_dir(out_dir)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=30, alpha=0.6, color='steelblue')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual - {model_name} Degree {degree} ({dataset_type})")
    
    # reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=2)
    ax.grid(True, alpha=0.3)
    
    filename = f"pred_vs_true_{model_name}_deg{degree}_{dataset_type}.png"
    out_path = os.path.join(out_dir, filename)
    _simple_save(fig, out_path)

def plot_residuals_detailed(y_true, y_pred, model_name, degree, dataset_type, out_dir="reports/figures"):
    """Plot residuals dengan nama file yang detail"""
    ensure_dir(out_dir)
    
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, residuals, s=30, alpha=0.6, color='coral')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals - {model_name} Degree {degree} ({dataset_type})")
    ax.grid(True, alpha=0.3)
    
    filename = f"residuals_{model_name}_deg{degree}_{dataset_type}.png"
    out_path = os.path.join(out_dir, filename)
    _simple_save(fig, out_path)

def plot_learning_curve_detailed(estimator, X, y, model_name, degree, cv=5, out_dir="reports/figures"):
    """Plot learning curve dengan nama file yang detail"""
    ensure_dir(out_dir)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve - {model_name} Degree {degree}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f"learning_curve_{model_name}_deg{degree}.png"
    out_path = os.path.join(out_dir, filename)
    _simple_save(fig, out_path)

def plot_coefficient_importance(model, feature_names, model_name, degree, out_dir="reports/figures"):
    """Plot coefficient importance untuk model linear"""
    ensure_dir(out_dir)
    
    # Ambil coefficients dari model
    if hasattr(model, 'named_steps') and 'reg' in model.named_steps:
        coef = model.named_steps['reg'].coef_
    else:
        coef = model.coef_
    
    # Jika ada terlalu banyak coefficients (polynomial features), ambil yang penting saja
    if len(coef) > 20:
        # Ambil 15 coefficient terbesar berdasarkan absolute value
        indices = np.argsort(np.abs(coef))[-15:]
        coef = coef[indices]
        feature_names = [f"Feature_{i}" for i in indices]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    colors = ['red' if c < 0 else 'blue' for c in coef]
    bars = ax.barh(range(len(coef)), coef, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(coef)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"Coefficient Importance - {model_name} Degree {degree}")
    ax.grid(True, alpha=0.3, axis='x')
    
    filename = f"coeff_importance_{model_name}_deg{degree}.png"
    out_path = os.path.join(out_dir, filename)
    _simple_save(fig, out_path)

# Fungsi lama untuk backward compatibility
def plot_pred_vs_actual(y_true, y_pred, out_path="reports/figures/pred_vs_actual.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=20)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    # reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims)
    _simple_save(fig, out_path)

def plot_residuals(y_true, y_pred, out_path="reports/figures/residuals.png"):
    residuals = y_true - y_pred
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, residuals, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals")
    _simple_save(fig, out_path)

def plot_learning_curve(estimator, X, y, cv, out_path="reports/figures/learning_curve.png"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.plot(train_sizes, val_mean, 'o-', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend()
    _simple_save(fig, out_path)

def plot_model_comparison(labels, scores, out_path="reports/figures/model_comparison_r2.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(labels, scores)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison")
    ax.tick_params(axis='x', rotation=45)
    _simple_save(fig, out_path)

def plot_learning_curve(estimator, X, y, cv, out_path="reports/figures/learning_curve.png"):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8)
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_sizes, train_mean, marker="o", label="Train R2")
    ax.plot(train_sizes, test_mean, marker="s", label="CV R2")
    ax.set_xlabel("Jumlah Sampel Latih")
    ax.set_ylabel("R^2")
    ax.set_title("Learning Curve")
    ax.legend()
    _simple_save(fig, out_path)

def plot_model_comparison(labels, scores, out_path="reports/figures/model_comparison_r2.png"):
    idx = np.arange(len(labels))
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.bar(idx, scores)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean CV R^2")
    ax.set_title("Perbandingan Model (Derajat & Regulator)")
    _simple_save(fig, out_path)