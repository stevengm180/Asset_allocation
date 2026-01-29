"""
Module de stacking ensemble - Meta-learning
Combine les prédictions des modèles avec un meta-modèle
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgbm


def create_stacking_features(cv_models, lgbm_final, X_train, X_val, X_test):
    """
    Crée des features de stacking à partir des prédictions.
    
    Parameters
    ----------
    cv_models : list
        Modèles LightGBM des folds CV
    lgbm_final : lgbm.Booster
        Modèle LightGBM entraîné sur toutes les données
    X_train, X_val, X_test : pd.DataFrame
        
    Returns
    -------
    tuple
        (X_train_meta, X_val_meta, X_test_meta)
    """
    print("🔗 Création des features de stacking...\n")
    
    n_folds = len(cv_models)
    
    # Train features
    X_train_meta = np.zeros((X_train.shape[0], n_folds + 1))
    for i, model in enumerate(cv_models):
        X_train_meta[:, i] = model.predict(X_train)
    X_train_meta[:, -1] = lgbm_final.predict(X_train)
    
    # Val features
    X_val_meta = np.zeros((X_val.shape[0], n_folds + 1))
    for i, model in enumerate(cv_models):
        X_val_meta[:, i] = model.predict(X_val)
    X_val_meta[:, -1] = lgbm_final.predict(X_val)
    
    # Test features
    X_test_meta = np.zeros((X_test.shape[0], n_folds + 1))
    for i, model in enumerate(cv_models):
        X_test_meta[:, i] = model.predict(X_test)
    X_test_meta[:, -1] = lgbm_final.predict(X_test)
    
    print(f"✓ Features de stacking créées: shape {X_train_meta.shape}")
    print(f"  {n_folds} folds CV + modèle final = {n_folds + 1} meta-features")
    
    return X_train_meta, X_val_meta, X_test_meta


def train_stacking_meta_model(X_train_meta, y_train, X_val_meta, y_val):
    """
    Entraîne le meta-modèle Ridge sur les features de stacking.
    
    Parameters
    ----------
    X_train_meta : np.ndarray
        Features de stacking train
    y_train : pd.Series
    X_val_meta : np.ndarray
        Features de stacking val
    y_val : pd.Series
        
    Returns
    -------
    Ridge, float
        Meta-modèle et accuracy
    """
    print("\n🎯 Entraînement du meta-modèle (Ridge)...\n")
    
    meta_model = Ridge(alpha=0.1)
    meta_model.fit(X_train_meta, y_train)
    
    # Évaluation
    y_train_pred = (meta_model.predict(X_train_meta) > 0.5).astype(int)
    y_val_pred = (meta_model.predict(X_val_meta) > 0.5).astype(int)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    
    return meta_model, val_acc


def generate_stacking_predictions(meta_model, X_test_meta):
    """
    Génère les prédictions avec le meta-modèle.
    
    Parameters
    ----------
    meta_model : Ridge
    X_test_meta : np.ndarray
        
    Returns
    -------
    np.ndarray
        Prédictions binaires
    """
    print("\n📊 Génération des prédictions avec stacking...\n")
    
    predictions_proba = meta_model.predict(X_test_meta)
    predictions = (predictions_proba > 0.5).astype(int)
    
    print(f"✓ Positif: {predictions.sum():,} ({predictions.mean():.2%})")
    
    return predictions, predictions_proba


def compare_stacking_vs_ensemble(predictions_simple_ensemble, predictions_stacking):
    """
    Compare les prédictions simples et stacking.
    """
    print("\n📈 Comparaison Ensemble Simple vs Stacking:\n")
    
    agreement = (predictions_simple_ensemble == predictions_stacking).mean()
    print(f"  Accord entre les deux: {agreement:.2%}")
    
    diff = np.abs(predictions_simple_ensemble.sum() - predictions_stacking.sum())
    print(f"  Différence de positifs: {diff} samples")
    
    return agreement


def multi_level_stacking(cv_models, lgbm_final, X_train, X_val, X_test, y_train, y_val):
    """
    Crée un stacking multi-niveau pour plus de stabilité.
    
    Parameters
    ----------
    cv_models : list
    lgbm_final : lgbm.Booster
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val : pd.Series
        
    Returns
    -------
    tuple
        (predictions, meta_model_accuracy)
    """
    print("🏔️ MULTI-LEVEL STACKING\n")
    
    # Level 1: Prédictions des modèles
    X_train_meta, X_val_meta, X_test_meta = create_stacking_features(
        cv_models, lgbm_final, X_train, X_val, X_test
    )
    
    # Level 2: Meta-modèle
    meta_model, meta_acc = train_stacking_meta_model(X_train_meta, y_train, X_val_meta, y_val)
    
    # Prédictions finales
    predictions, predictions_proba = generate_stacking_predictions(meta_model, X_test_meta)
    
    return predictions, meta_acc, meta_model


def weighted_ensemble(cv_models, lgbm_final, X_test, cv_scores):
    """
    Crée un ensemble pondéré basé sur les performances CV.
    
    Parameters
    ----------
    cv_models : list
    lgbm_final : lgbm.Booster
    X_test : pd.DataFrame
    cv_scores : list
        Scores de chaque fold
        
    Returns
    -------
    np.ndarray
        Prédictions pondérées
    """
    print("⚖️ WEIGHTED ENSEMBLE (basé sur CV scores)\n")
    
    # Normaliser les poids
    weights = np.array(cv_scores)
    weights = weights / weights.sum()
    
    print(f"  Poids CV: {[f'{w:.3f}' for w in weights]}\n")
    
    # Prédictions pondérées
    ensemble_pred = np.zeros(X_test.shape[0])
    
    for i, (model, weight) in enumerate(zip(cv_models, weights)):
        pred = model.predict(X_test)
        ensemble_pred += weight * pred
        print(f"  Fold {i+1}: poids={weight:.3f}")
    
    # Ajouter le modèle final (poids égal à la moyenne des CV)
    final_weight = weights.mean()
    ensemble_pred += final_weight * lgbm_final.predict(X_test)
    print(f"  Final: poids={final_weight:.3f}")
    
    # Renormaliser
    n_models = len(cv_models) + 1
    ensemble_pred = ensemble_pred / n_models
    
    predictions = (ensemble_pred > 0.5).astype(int)
    
    print(f"\n✓ Positif: {predictions.sum():,} ({predictions.mean():.2%})")
    
    return predictions
