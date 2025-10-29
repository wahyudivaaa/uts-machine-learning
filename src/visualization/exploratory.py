import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from src.utils.io import ensure_dir

def create_histograms(df, target_col='y', output_dir="reports/figures"):
    """Membuat histogram untuk setiap feature"""
    ensure_dir(output_dir)
    
    # Mapping nama kolom ke nama yang lebih deskriptif
    feature_names = {
        'x1': 'AksesTransport',
        'x2': 'Harga', 
        'x3': 'JarakPusat',
        'x4': 'Kamar',
        'x5': 'KamarMandi'
    }
    
    # Tambahkan kolom target juga
    if target_col not in feature_names:
        feature_names[target_col] = 'Luas'
    
    for col in df.columns:
        if col != target_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(feature_names.get(col, col))
            ax.set_ylabel('Frekuensi')
            ax.set_title(f'Distribusi {feature_names.get(col, col)}')
            ax.grid(True, alpha=0.3)
            
            filename = f"hist_{feature_names.get(col, col)}.png"
            filepath = os.path.join(output_dir, filename)
            fig.tight_layout()
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

def create_scatter_plots(df, target_col='y', output_dir="reports/figures"):
    """Membuat scatter plot antara setiap feature dengan target"""
    ensure_dir(output_dir)
    
    # Mapping nama kolom ke nama yang lebih deskriptif
    feature_names = {
        'x1': 'AksesTransport',
        'x2': 'Harga',
        'x3': 'JarakPusat', 
        'x4': 'Kamar',
        'x5': 'KamarMandi'
    }
    
    target_name = 'Harga'  # Target adalah harga
    
    for col in df.columns:
        if col != target_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df[col], df[target_col], alpha=0.6, color='coral', s=30)
            ax.set_xlabel(feature_names.get(col, col))
            ax.set_ylabel(target_name)
            ax.set_title(f'{feature_names.get(col, col)} vs {target_name}')
            ax.grid(True, alpha=0.3)
            
            filename = f"scatter_{feature_names.get(col, col)}_vs_{target_name}.png"
            filepath = os.path.join(output_dir, filename)
            fig.tight_layout()
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

def create_correlation_heatmap(df, output_dir="reports/figures"):
    """Membuat correlation heatmap"""
    ensure_dir(output_dir)
    
    # Hitung korelasi
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap')
    
    filepath = os.path.join(output_dir, "corr_heatmap.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_all_exploratory_plots(csv_path="data/raw/synthetic_regression.csv", output_dir="reports/figures"):
    """Membuat semua plot eksploratori"""
    df = pd.read_csv(csv_path)
    
    print("Membuat histogram...")
    create_histograms(df, output_dir=output_dir)
    
    print("Membuat scatter plots...")
    create_scatter_plots(df, output_dir=output_dir)
    
    print("Membuat correlation heatmap...")
    create_correlation_heatmap(df, output_dir=output_dir)
    
    print("Semua plot eksploratori selesai dibuat!")