"""
Module de cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from config import N_SPLITS_CV, CV_RANDOM_STATE, CV_SHUFFLE, LGBM_PARAMS, LGBM_BOOSTING_ROUNDS
from utils import plot_cv_scores


def perform_time_series_cross_validation(X, y, train_data, all_features):
    """
    Effectue une cross-validation respectueuse de la temporalité des données.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features (inclus TS)
    y : pd.Series
        Cible
    train_data : pd.DataFrame
        Données originales avec TS
    all_features : list
        Noms des features
        
    Returns
    -------
    tuple
        (cv_scores, cv_models, mean_score, std_score)
    """
    print("🔄 Cross-Validation LightGBM avec Time Series Split...\n")
    
    train_dates = train_data['TS'].unique()
    n_splits = N_SPLITS_CV
    
    cv_scores = []
    cv_models = []
    
    splits = KFold(
        n_splits=n_splits, 
        random_state=CV_RANDOM_STATE,
        shuffle=CV_SHUFFLE
    ).split(train_dates)
    
    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates = train_dates[local_test_dates_ids]
        
        local_train_ids = train_data['TS'].isin(local_train_dates)
        local_test_ids = train_data['TS'].isin(local_test_dates)
        
        X_local_train = X.loc[local_train_ids, all_features]
        y_local_train = y.loc[local_train_ids]
        X_local_test = X.loc[local_test_ids, all_features]
        y_local_test = y.loc[local_test_ids]
        
        # Entraînement
        train_data_lgbm = lgbm.Dataset(X_local_train, label=y_local_train.values)
        model_lgbm = lgbm.train(LGBM_PARAMS, train_data_lgbm, num_boost_round=LGBM_BOOSTING_ROUNDS)
        
        # Prédiction
        y_local_pred = model_lgbm.predict(X_local_test.values, num_threads=LGBM_PARAMS['num_threads'])
        
        # Évaluation
        cv_models.append(model_lgbm)
        score = accuracy_score(
            (y_local_test > 0).astype(int),
            (y_local_pred > 0.5).astype(int)
        )
        cv_scores.append(score)
        
        print(f"Fold {i+1}/{n_splits} - Accuracy: {score:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\n✓ Cross-Validation terminée")
    print(f"Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Confidence Interval: [{mean_score - std_score:.4f}, {mean_score + std_score:.4f}]")
    
    return cv_scores, cv_models, mean_score, std_score


def plot_cv_results(cv_scores, mean_score, std_score):
    """
    Affiche les résultats de la cross-validation.
    """
    n_splits = len(cv_scores)
    
    fig, ax = plot_cv_scores(cv_scores, mean_score, std_score)
    plt.show()
    
    return fig, ax


def get_feature_importance(cv_models, all_features, top_n=30):
    """
    Extrait l'importance moyenne des features sur tous les modèles CV.
    
    Parameters
    ----------
    cv_models : list
        Liste des modèles LightGBM entraînés
    all_features : list
    top_n : int
        
    Returns
    -------
    pd.DataFrame
        Importance des features triée
    """
    print("📊 Extraction de l'importance des features...\n")
    
    feature_importances = []
    
    for model in cv_models:
        importance = model.feature_importance(importance_type='gain')
        feature_importances.append(importance)
    
    mean_importance = np.mean(feature_importances, axis=0)
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': mean_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"Top {top_n} Features:\n")
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    return feature_importance_df


def analyze_cv_performance(cv_scores):
    """
    Analyse complète de la performance CV.
    """
    print("\n📈 ANALYSE CV:")
    print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean: {np.mean(cv_scores):.4f}")
    print(f"  Std: {np.std(cv_scores):.4f}")
    print(f"  Min: {np.min(cv_scores):.4f}")
    print(f"  Max: {np.max(cv_scores):.4f}")
    print(f"  Range: {np.max(cv_scores) - np.min(cv_scores):.4f}")
    
    return {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'min': np.min(cv_scores),
        'max': np.max(cv_scores),
        'scores': cv_scores
    }
