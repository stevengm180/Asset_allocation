"""
Module d'entraînement des modèles
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
from sklearn.metrics import accuracy_score

from config import RIDGE_ALPHA, LOGISTIC_MAX_ITER, RF_N_ESTIMATORS, RF_MAX_DEPTH, LGBM_PARAMS, LGBM_BOOSTING_ROUNDS
from utils import calculate_metrics, print_metrics


def train_ridge(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle Ridge Regression.
    """
    print("🔵 Entraînement Ridge Regression...")
    
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train, y_train)
    
    y_train_pred = (ridge.predict(X_train) > 0.5).astype(int)
    y_val_pred = (ridge.predict(X_val) > 0.5).astype(int)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}\n")
    
    return ridge, train_acc, val_acc


def train_logistic(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle Logistic Regression.
    """
    print("🔴 Entraînement Logistic Regression...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logistic = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=42, n_jobs=1)
    logistic.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, logistic.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, logistic.predict(X_val_scaled))
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}\n")
    
    return logistic, train_acc, val_acc, scaler


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle Random Forest.
    """
    print("🟠 Entraînement Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, 
        max_depth=RF_MAX_DEPTH,
        random_state=42, 
        n_jobs=1
    )
    rf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}\n")
    
    return rf, train_acc, val_acc


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle LightGBM.
    """
    print("🟢 Entraînement LightGBM...")
    
    train_data = lgbm.Dataset(X_train, label=y_train)
    lgbm_model = lgbm.train(
        LGBM_PARAMS, 
        train_data, 
        num_boost_round=LGBM_BOOSTING_ROUNDS,
        valid_names=['train']
    )
    
    y_train_pred = (lgbm_model.predict(X_train) > 0.5).astype(int)
    y_val_pred = (lgbm_model.predict(X_val) > 0.5).astype(int)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}\n")
    
    return lgbm_model, train_acc, val_acc


def train_base_models(X_train_split, y_train_split, X_val_split, y_val_split):
    """
    Entraîne tous les modèles de base.
    
    Returns
    -------
    dict
        Résultats de tous les modèles
    """
    print("🔄 Entraînement des modèles de base...\n")
    
    results = {}
    
    # Ridge
    ridge, ridge_train, ridge_val = train_ridge(X_train_split, y_train_split, X_val_split, y_val_split)
    results['Ridge'] = {'model': ridge, 'train_acc': ridge_train, 'val_acc': ridge_val}
    
    # Logistic
    logistic, log_train, log_val, scaler = train_logistic(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    results['LogisticRegression'] = {'model': logistic, 'train_acc': log_train, 'val_acc': log_val, 'scaler': scaler}
    
    # Random Forest
    rf, rf_train, rf_val = train_random_forest(X_train_split, y_train_split, X_val_split, y_val_split)
    results['RandomForest'] = {'model': rf, 'train_acc': rf_train, 'val_acc': rf_val}
    
    # LightGBM
    lgbm_model, lgbm_train, lgbm_val = train_lightgbm(X_train_split, y_train_split, X_val_split, y_val_split)
    results['LightGBM'] = {'model': lgbm_model, 'train_acc': lgbm_train, 'val_acc': lgbm_val}
    
    return results


def create_results_dataframe(results):
    """
    Crée un DataFrame de résumé des résultats.
    """
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train Accuracy': [results[m]['train_acc'] for m in results.keys()],
        'Val Accuracy': [results[m]['val_acc'] for m in results.keys()]
    })
    
    print("\n📊 RÉSUMÉ DES MODÈLES DE BASE:\n")
    print(summary_df.to_string(index=False))
    
    best_idx = summary_df['Val Accuracy'].idxmax()
    best_model = summary_df.loc[best_idx, 'Model']
    best_acc = summary_df.loc[best_idx, 'Val Accuracy']
    
    print(f"\n✨ Meilleur modèle: {best_model} (Accuracy: {best_acc:.4f})")
    
    return summary_df
