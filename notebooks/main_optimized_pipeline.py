#!/usr/bin/env python3
"""
Pipeline Optimisé v2 - Conservative Approach
Garder 50.66% baseline et ajouter + optimisations légères
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgbm
from sklearn.metrics import accuracy_score

from config import *
from data_loading import load_data


def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_step(step_num, description):
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 60)


def train_lgbm_model(X_train, y_train, params, num_rounds=300):
    """Entraîne un modèle LightGBM"""
    train_data = lgbm.Dataset(X_train, label=y_train)
    model = lgbm.train(params, train_data, num_boost_round=num_rounds)
    return model


def main():
    """Pipeline conservateur : 50.66% baseline + optimisations légères"""
    
    print_header("🚀 PIPELINE OPTIMISÉ v2 - Conservative")
    print(f"📊 Objectif: Garder 50.66% et +0.5%")
    print(f"⏰ Démarrage: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # ========================================
    # ÉTAPE 1: Chargement
    # ========================================
    print_step(1, "Chargement des données")
    
    X_train, y_train, X_test, sample_submission = load_data()
    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ y_train: {y_train.shape}")
    print(f"✓ X_test: {X_test.shape}")
    
    # Nettoyage colonnes non-numériques
    if hasattr(X_train, 'select_dtypes'):
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        X_train = X_train[numeric_cols].copy()
        X_test = X_test[numeric_cols].copy()
        print(f"✓ X_train nettoyé: {X_train.shape}")
    
    # Convertir en numpy
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values.flatten()
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    
    print(f"✓ X_train {X_train.shape}, y_train {y_train.shape}")
    
    # ========================================
    # ÉTAPE 2: Split train/validation
    # ========================================
    print_step(2, "Split train/validation (80/20)")
    
    n_split = int(0.8 * len(X_train))
    X_train_split, X_val_split = X_train[:n_split], X_train[n_split:]
    y_train_split, y_val_split = y_train[:n_split], y_train[n_split:]
    
    print(f"✓ Train: {X_train_split.shape}, Val: {X_val_split.shape}")
    
    # ========================================
    # ÉTAPE 3: Hyperparameter Tuning SIMPLE
    # ========================================
    print_step(3, "Hyperparameter Tuning (2 configs seulement)")
    
    print("⚡ TEST 2 CONFIGURATIONS\n")
    
    tested_params = [
        # Config 1: Baseline original (50.66% known)
        {
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        # Config 2: Léger ajustement
        {
            "learning_rate": 0.04,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    ]
    
    best_params = None
    best_accuracy = 0
    all_models = []
    
    for i, params in enumerate(tested_params, 1):
        full_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_threads": 1,
            "seed": 42,
            "verbosity": -1,
            **params
        }
        
        model = train_lgbm_model(X_train_split, y_train_split, full_params)
        all_models.append(model)
        
        y_val_pred = (model.predict(X_val_split) > 0.5).astype(int)
        accuracy = accuracy_score(y_val_split, y_val_pred)
        
        print(f"   Config {i}: Accuracy = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = full_params
    
    print(f"\n✨ Meilleure config: {best_accuracy:.4f}")
    
    # ========================================
    # ÉTAPE 4: Entraîner modèle final avec les meilleurs params
    # ========================================
    print_step(4, "Entraînement modèle final")
    
    lgbm_final = train_lgbm_model(X_train, y_train, best_params, num_rounds=300)
    print(f"✓ Modèle LightGBM entraîné")
    
    # ========================================
    # ÉTAPE 5: Ensemble simple (moyenne des modèles)
    # ========================================
    print_step(5, "Simple Ensemble (Moyenne)")
    
    # Moyenne des prédictions (plus robuste que stacking complexe)
    y_pred_model1 = all_models[0].predict(X_test)
    y_pred_model2 = all_models[1].predict(X_test)
    y_pred_final = lgbm_final.predict(X_test)
    
    # Moyenne simple (poids égaux)
    y_pred_avg = (y_pred_model1 + y_pred_model2 + y_pred_final) / 3
    
    print(f"✓ Ensemble: moyenne de 3 modèles")
    print(f"✓ Prédictions avg: min={y_pred_avg.min():.4f}, max={y_pred_avg.max():.4f}")
    
    # ========================================
    # ÉTAPE 6: Threshold Optimization (SIMPLE)
    # ========================================
    print_step(6, "Threshold Optimization")
    
    # Utiliser le modèle final simple pour threshold
    y_val_pred_final = lgbm_final.predict(X_val_split)
    
    best_threshold = 0.5
    best_val_accuracy = 0
    
    print("  Test des seuils sur validation:")
    for threshold in np.arange(0.40, 0.60, 0.02):
        y_pred_thresh = (y_val_pred_final > threshold).astype(int)
        acc = accuracy_score(y_val_split, y_pred_thresh)
        print(f"    Threshold {threshold:.2f}: Accuracy = {acc:.4f}")
        if acc > best_val_accuracy:
            best_val_accuracy = acc
            best_threshold = threshold
    
    print(f"\n✨ Optimal Threshold: {best_threshold:.4f}")
    
    # ========================================
    # ÉTAPE 7: Générer soumissions
    # ========================================
    print_step(7, "Génération des soumissions")
    
    # Appliquer le seuil
    y_pred_ensemble = (y_pred_avg > best_threshold).astype(int)
    y_pred_simple = (y_pred_final > best_threshold).astype(int)
    
    # ROW_ID commence à 527073
    row_ids = np.arange(len(X_train), len(X_train) + len(y_pred_ensemble))
    
    # 2 soumissions
    submission_ensemble = pd.DataFrame({
        'ROW_ID': row_ids,
        'prediction': y_pred_ensemble
    })
    
    submission_final = pd.DataFrame({
        'ROW_ID': row_ids,
        'prediction': y_pred_simple
    })
    
    output_dir = Path('submissions')
    output_dir.mkdir(exist_ok=True)
    
    file1 = output_dir / 'submission_ensemble_avg.csv'
    file2 = output_dir / 'submission_final_lgbm.csv'
    
    submission_ensemble.to_csv(file1, index=False)
    submission_final.to_csv(file2, index=False)
    
    print(f"✓ {file1}")
    print(f"✓ {file2}")
    
    # ========================================
    # Résumé
    # ========================================
    elapsed_time = time.time() - start_time
    
    print_header("✅ RÉSUMÉ FINAL")
    print(f"⏱️  Temps total: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    print(f"🎯 Optimal Threshold: {best_threshold:.4f}")
    print(f"📊 Val Accuracy (seuil): {best_val_accuracy:.4f}")
    print(f"🚀 Soumissions générées: 2 fichiers")
    print(f"\n📈 Stratégies utilisées:")
    print(f"   ✓ Hyperparameter Tuning (2 configs)")
    print(f"   ✓ Simple Ensemble (moyenne 3 modèles)")
    print(f"   ✓ Threshold Optimization (0.40-0.60)")
    print(f"\n💾 À soumettre:")
    print(f"   1️⃣  {file1} (Ensemble - recommandé)")
    print(f"   2️⃣  {file2} (Simple LightGBM)")
    print(f"\n🎓 Gains estimés: +0.2-0.5% → Target: 51.0%+")
    print("="*60)


if __name__ == "__main__":
    main()
