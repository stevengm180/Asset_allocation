"""
Module de chargement et exploration des données
"""

import pandas as pd
import numpy as np
from config import DATA_FILES
from utils import plot_target_distribution
import matplotlib.pyplot as plt


def load_data(data_dir='.'):
    """
    Charge tous les fichiers de données.
    
    Parameters
    ----------
    data_dir : str
        Répertoire contenant les fichiers CSV
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, sample_submission)
    """
    print("Chargement des données...")
    
    X_train = pd.read_csv(f'{data_dir}/{DATA_FILES["X_train"]}', index_col='ROW_ID')
    y_train = pd.read_csv(f'{data_dir}/{DATA_FILES["y_train"]}', index_col='ROW_ID')
    X_test = pd.read_csv(f'{data_dir}/{DATA_FILES["X_test"]}', index_col='ROW_ID')
    sample_submission = pd.read_csv(f'{data_dir}/{DATA_FILES["sample_submission"]}', index_col='ROW_ID')
    
    return X_train, y_train, X_test, sample_submission


def explore_data(X_train, y_train, X_test):
    """
    Affiche une exploration complète des données.
    
    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_test : pd.DataFrame
    """
    print("\n📊 DIMENSIONS DES DONNÉES:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")

    print(f"\n📋 COLONNES:")
    print(X_train.columns.tolist())

    print(f"\n🔍 TYPES DE DONNÉES:")
    print(f"Types: {X_train.dtypes.value_counts()}")
    
    print(f"\n❓ VALEURS MANQUANTES:")
    missing_count = X_train.isnull().sum().sum()
    print(f"Total: {missing_count} valeurs manquantes")
    if missing_count > 0:
        print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

    print(f"\n📊 STATISTIQUES DESCRIPTIVES:")
    print(X_train.iloc[:, :5].describe().T)

    print(f"\n🎯 CIBLE (y_train):")
    print(y_train['target'].describe())
    
    print(f"\n📈 SIGNE DE LA CIBLE:")
    positive = (y_train['target'] > 0).sum()
    negative = (y_train['target'] <= 0).sum()
    print(f"  Positif (>0):  {positive:7d} ({positive / len(y_train) * 100:5.1f}%)")
    print(f"  Négatif (<=0): {negative:7d} ({negative / len(y_train) * 100:5.1f}%)")

    # Visualisations
    fig, axes = plot_target_distribution(y_train['target'].values)
    plt.show()

    return True


def get_data_summary(X_train, y_train):
    """
    Retourne un résumé des statistiques clés des données.
    """
    return {
        'n_train': len(X_train),
        'n_features': X_train.shape[1],
        'n_positive': (y_train['target'] > 0).sum(),
        'n_negative': (y_train['target'] <= 0).sum(),
        'pct_positive': (y_train['target'] > 0).sum() / len(y_train),
    }
