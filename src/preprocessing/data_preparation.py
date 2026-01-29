"""
Module de préparation et nettoyage des données
"""

import pandas as pd
import numpy as np


def prepare_data(X_train, y_train, X_test):
    """
    Fusionne et prépare les données d'entraînement.
    
    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_test : pd.DataFrame
        
    Returns
    -------
    tuple
        (train_data, X_test_clean)
    """
    print("Préparation des données...\n")
    
    # Fusion
    train_data = X_train.copy()
    train_data['target'] = y_train['target'].values
    
    print(f"✓ Fusion complète: {train_data.shape}")

    # Traitement des valeurs manquantes
    missing_train = train_data.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    
    if missing_train > 0 or missing_test > 0:
        print(f"\nTraitement des valeurs manquantes:")
        print(f"  Train: {missing_train} → 0 (remplissage)")
        print(f"  Test: {missing_test} → 0 (remplissage)")
        
        train_data = train_data.fillna(0)
        X_test = X_test.fillna(0)

    print("✓ Données préparées et nettoyées")
    
    return train_data, X_test


def handle_outliers_detection(train_data, features, n_std=3):
    """
    Détecte les outliers en utilisant la méthode IQR.
    
    Parameters
    ----------
    train_data : pd.DataFrame
    features : list
        Liste des colonnes à vérifier
    n_std : float
        Nombre d'écarts-types pour définir les seuils
        
    Returns
    -------
    dict
        Statistiques sur les outliers par feature
    """
    outliers_summary = {}
    
    for col in features:
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = ((train_data[col] < Q1 - n_std*IQR) | 
                       (train_data[col] > Q3 + n_std*IQR))
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            outliers_summary[col] = {
                'count': n_outliers,
                'percentage': n_outliers / len(train_data) * 100
            }
    
    return outliers_summary


def detect_and_report_outliers(train_data, base_features):
    """
    Détecte et affiche un rapport sur les outliers.
    
    Parameters
    ----------
    train_data : pd.DataFrame
    base_features : list
        Colonnes à vérifier
    """
    print("\nDétection des outliers (± 3 écarts-types):")
    
    outliers_info = handle_outliers_detection(train_data, base_features[:5])
    
    if outliers_info:
        for feat, info in outliers_info.items():
            print(f"  {feat}: {info['count']} outliers ({info['percentage']:.2f}%)")
    else:
        print("  Aucun outlier détecté")
    
    return outliers_info


def split_temporal_data(train_data, train_test_split_ratio=0.7):
    """
    Effectue un split temporel respectueux de la structure time series.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Doit contenir une colonne 'TS'
    train_test_split_ratio : float
        Proportion pour l'entraînement
        
    Returns
    -------
    tuple
        (X_train_split, y_train_split, X_val_split, y_val_split)
    """
    train_dates = train_data['TS'].unique()
    n_train_dates = int(len(train_dates) * train_test_split_ratio)
    train_date_split = train_dates[n_train_dates]

    train_mask = train_data['TS'] < train_date_split
    test_mask = train_data['TS'] >= train_date_split

    print(f"\n📊 Split temporel:")
    print(f"  Date de séparation: {train_date_split}")
    print(f"  Train: {train_mask.sum():,} samples")
    print(f"  Val: {test_mask.sum():,} samples")
    
    return train_mask, test_mask


def prepare_features_for_modeling(train_data, X_test, features):
    """
    Prépare les features pour la modélisation.
    
    Parameters
    ----------
    train_data : pd.DataFrame
    X_test : pd.DataFrame
    features : list
        Noms des colonnes de features
        
    Returns
    -------
    tuple
        (X, y, X_test_clean)
    """
    X = train_data[features].fillna(0)
    y = (train_data['target'] > 0).astype(int)
    X_test_clean = X_test[features].fillna(0)
    
    print(f"\n✓ Features préparées:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Classes: {y.value_counts().to_dict()}")
    
    return X, y, X_test_clean
