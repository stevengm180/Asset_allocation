"""
Module d'optimisation des hyperparamètres avec Optuna
Trouve les meilleurs hyperparamètres pour LightGBM
"""

import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')


def objective_lgbm_cv(trial, X_train, y_train, X_val, y_val):
    """
    Fonction objective pour Optuna - optimise LightGBM
    """
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 1,
        "seed": 42,
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }
    
    train_data = lgbm.Dataset(X_train, label=y_train)
    model = lgbm.train(params, train_data, num_boost_round=300)
    
    y_val_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    return accuracy


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimise les hyperparamètres LightGBM avec Optuna.
    
    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
    X_val, y_val : pd.DataFrame, pd.Series
    n_trials : int
        Nombre de trials Optuna
        
    Returns
    -------
    dict
        Meilleurs paramètres trouvés
    """
    print("🔍 OPTIMISATION HYPERPARAMÈTRES LIGHTGBM\n")
    print(f"   Nombre de trials: {n_trials}")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Val set: {X_val.shape[0]} samples\n")
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="lgbm_optimization"
    )
    
    # Optimisation
    study.optimize(
        lambda trial: objective_lgbm_cv(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Résultats
    best_trial = study.best_trial
    
    print(f"\n✨ MEILLEUR TRIAL: {best_trial.number}")
    print(f"   Accuracy: {best_trial.value:.4f}\n")
    print("   Hyperparamètres:")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")
    
    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 1,
        "seed": 42,
        "verbosity": -1,
        **best_trial.params
    }
    
    return best_params


def train_lgbm_optimized(X_train, y_train, X_val, y_val, best_params, num_rounds=500):
    """
    Entraîne LightGBM avec les paramètres optimisés.
    
    Parameters
    ----------
    best_params : dict
        Hyperparamètres optimisés
    num_rounds : int
        Nombre de boosting rounds
        
    Returns
    -------
    lgbm.Booster, float
        Modèle et accuracy
    """
    print(f"\n🚀 Entraînement LightGBM avec hyperparamètres optimisés...")
    
    train_data = lgbm.Dataset(X_train, label=y_train)
    model = lgbm.train(best_params, train_data, num_boost_round=num_rounds, verbose_eval=0)
    
    y_val_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"✓ Val Accuracy: {accuracy:.4f}")
    
    return model, accuracy


def quick_hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Version rapide: teste simplement les meilleurs paramètres connus.
    """
    print("⚡ QUICK HYPERPARAMETER SEARCH (pas d'optimisation)\n")
    
    # Paramètres recommandés pour ce type de problème
    tested_params = [
        {
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
        },
        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 63,
            "min_child_samples": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "learning_rate": 0.08,
            "max_depth": 7,
            "num_leaves": 95,
            "min_child_samples": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
    ]
    
    best_params = None
    best_accuracy = 0
    
    for i, params in enumerate(tested_params, 1):
        full_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_threads": 1,
            "seed": 42,
            "verbosity": -1,
            **params
        }
        
        train_data = lgbm.Dataset(X_train, label=y_train)
        model = lgbm.train(full_params, train_data, num_boost_round=300)
        
        y_val_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"   Config {i}: Accuracy = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = full_params
    
    print(f"\n✨ Meilleur config: {best_accuracy:.4f}")
    return best_params
