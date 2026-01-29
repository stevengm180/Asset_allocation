"""
Module de génération des prédictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgbm

from config import LGBM_PARAMS, LGBM_BOOSTING_ROUNDS, SUBMISSION_FILES
from model_training import train_ridge


def train_final_model(X, y, all_features):
    """
    Entraîne le modèle final sur toutes les données.
    
    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    all_features : list
        
    Returns
    -------
    lgbm.Booster
    """
    print("🚀 Entraînement du modèle final sur toutes les données...\n")
    
    train_data_final = lgbm.Dataset(X[all_features], label=y)
    model_final = lgbm.train(LGBM_PARAMS, train_data_final, num_boost_round=LGBM_BOOSTING_ROUNDS)
    
    print("✓ Modèle final entraîné")
    
    return model_final


def generate_predictions(model_final, cv_models, X_test_features):
    """
    Génère les prédictions finales sur le test set.
    
    Parameters
    ----------
    model_final : lgbm.Booster
        Modèle entraîné sur toutes les données
    cv_models : list
        Modèles de la cross-validation
    X_test_features : pd.DataFrame
        Features du test set
        
    Returns
    -------
    tuple
        (predictions_final, predictions_ensemble)
    """
    print(f"\n📊 Génération des prédictions sur {X_test_features.shape[0]} samples...\n")
    
    # Prédictions du modèle final
    predictions_raw_final = model_final.predict(X_test_features)
    predictions_final = (predictions_raw_final > 0.5).astype(int)
    
    print(f"✓ Modèle Final LightGBM:")
    print(f"  Distribution: {np.bincount(predictions_final)}")
    print(f"  Proportion positif: {predictions_final.mean():.2%}")
    
    # Ensemble (moyenne des CV folds)
    if not cv_models:
        print("\n⚠️  Aucun modèle CV fourni. Ensemble = modèle final.")
        predictions_ensemble = predictions_final.copy()
    else:
        ensemble_predictions = np.zeros(len(X_test_features))
        for i, model in enumerate(cv_models):
            pred = model.predict(X_test_features)
            ensemble_predictions += pred
            if i == 0:
                print(f"\n✓ Ensemble (moyenne {len(cv_models)} folds):")
        
        ensemble_predictions /= len(cv_models)
        predictions_ensemble = (ensemble_predictions > 0.5).astype(int)
        
        print(f"  Distribution: {np.bincount(predictions_ensemble)}")
        print(f"  Proportion positif: {predictions_ensemble.mean():.2%}")
    
    return predictions_final, predictions_ensemble


def save_submissions(predictions_dict, sample_submission):
    """
    Sauvegarde tous les fichiers de soumission.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionnaire {nom: array de prédictions}
    sample_submission : pd.DataFrame
        Template de soumission
        
    Returns
    -------
    dict
        Information sur les soumissions sauvegardées
    """
    print("\n💾 Sauvegarde des fichiers de soumission...\n")
    
    submissions_info = {}
    
    for name, predictions in predictions_dict.items():
        submission = sample_submission.copy()
        submission['prediction'] = predictions
        filename = SUBMISSION_FILES.get(name, f'submission_{name}.csv')
        # Ensure format matches sample_submission: columns [ROW_ID, prediction]
        submission.reset_index().to_csv(filename, index=False)
        
        submissions_info[name] = {
            'filename': filename,
            'positive': predictions.sum(),
            'percentage': predictions.mean()
        }
        
        print(f"✓ {filename}")
        print(f"  Positif: {predictions.sum():,} ({predictions.mean():.2%})")
    
    return submissions_info


def compare_predictions(predictions_dict):
    """
    Compare les prédictions des différents modèles.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionnaire {nom: array de prédictions}
        
    Returns
    -------
    pd.DataFrame
    """
    print("\n📈 Comparaison des prédictions:\n")
    
    comparison = pd.DataFrame(predictions_dict)
    
    print("Premiers 10 exemples:")
    print(comparison.head(10).to_string())
    
    print("\n\nAccord entre modèles:")
    model_names = list(predictions_dict.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            agreement = (predictions_dict[model_names[i]] == predictions_dict[model_names[j]]).mean()
            print(f"  {model_names[i]} vs {model_names[j]}: {agreement:.2%}")
    
    return comparison


def generate_predictions_with_baseline(X, y, X_test_features, model_final, cv_models, sample_submission):
    """
    Génère les prédictions avec la baseline Ridge.
    
    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    X_test_features : pd.DataFrame
    model_final : lgbm.Booster
    cv_models : list
    sample_submission : pd.DataFrame
        
    Returns
    -------
    dict
        Toutes les prédictions
    """
    # Prédictions LightGBM
    pred_final, pred_ensemble = generate_predictions(model_final, cv_models, X_test_features)
    
    # Ridge baseline
    print(f"\n🔵 Ridge Baseline:")
    ridge_model, _, _ = train_ridge(X, y, X_test_features, y[:len(X_test_features)])  # Dummy
    ridge_pred = (ridge_model.predict(X_test_features) > 0.5).astype(int)
    print(f"  Positif: {ridge_pred.sum():,} ({ridge_pred.mean():.2%})")
    
    predictions = {
        'lgbm_final': pred_final,
        'ensemble': pred_ensemble,
        'ridge_baseline': ridge_pred
    }
    
    # Sauvegarde
    submissions_info = save_submissions(predictions, sample_submission)
    
    # Comparaison
    compare_predictions(predictions)
    
    return predictions, submissions_info
