# Structure Modulaire - Prédiction des Allocations d'Actifs

## 📂 Architecture

La structure a été découpée en **modules Python indépendants et réutilisables** pour faciliter la maintenance, le testing et les extensions futures.

```
allocation_actifs/
├── config.py                     # Configuration centralisée
├── utils.py                      # Fonctions utilitaires (metrics, plots, reports)
├── data_loading.py               # Chargement et exploration des données
├── data_preparation.py           # Nettoyage, fusion, préparation
├── feature_engineering.py        # Création des features
├── exploratory_analysis.py       # Analyse exploratoire
├── model_training.py             # Entraînement des modèles
├── cross_validation.py           # Cross-validation et importance
├── predictions.py                # Génération des prédictions
├── main_pipeline.ipynb           # Notebook orchestrateur (à exécuter)
├── research_structure.ipynb      # Ancien notebook (référence)
└── README.md                     # Ce fichier
```

## 🔧 Description des Modules

### 1. **config.py**
Configuration centralisée de tous les hyperparamètres et chemins de fichiers.
- Colonnes de base (RET, SIGNED_VOLUME, etc.)
- Horizons de features
- Paramètres LightGBM, Ridge, Random Forest
- Chemins des fichiers

**Utilité**: Modifier rapidement les hyperparamètres sans toucher au code.

### 2. **utils.py**
Fonctions utilitaires réutilisables.
- `calculate_metrics()` - Calcul des métriques
- `plot_model_comparison()` - Visualisation comparative
- `plot_feature_importance()` - Graphique importance
- `plot_cv_scores()` - Evolution CV
- `print_summary_report()` - Résumé exécutif

### 3. **data_loading.py**
Gestion du chargement et exploration des données.
- `load_data()` - Charge CSV
- `explore_data()` - Statistiques descriptives
- `get_data_summary()` - Résumé clé

### 4. **data_preparation.py**
Nettoyage et préparation des données.
- `prepare_data()` - Fusion X_train + y_train
- `handle_outliers_detection()` - Détection outliers
- `split_temporal_data()` - Split respectueux temporalité
- `prepare_features_for_modeling()` - Prépare X, y pour modèles

### 5. **feature_engineering.py**
Ingénierie des features (+50 features créées).
- Moyennes mobiles (3, 5, 10, 15, 20j)
- Volatilité (std, skew, kurtosis)
- Features de momentum
- Features de volume
- Interactions
- Agrégations par groupe

### 6. **exploratory_analysis.py**
Analyse exploratoire des features.
- `analyze_correlations()` - Corrélation avec cible
- `analyze_features_by_class()` - Comparaison positif/négatif
- `get_feature_statistics()` - Stats complètes

### 7. **model_training.py**
Entraînement des modèles.
- `train_ridge()` - Ridge Regression
- `train_logistic()` - Logistic Regression
- `train_random_forest()` - Random Forest
- `train_lightgbm()` - LightGBM
- `train_base_models()` - Lance tous les modèles
- `create_results_dataframe()` - Résumé résultats

### 8. **cross_validation.py**
Cross-validation et analyse de performance.
- `perform_time_series_cross_validation()` - CV respectueuse temporalité
- `plot_cv_results()` - Visualisation CV
- `get_feature_importance()` - Importance moyenne des folds
- `analyze_cv_performance()` - Stats CV

### 9. **predictions.py**
Génération des prédictions finales.
- `train_final_model()` - Modèle final sur all data
- `generate_predictions()` - Prédictions sur test set
- `save_submissions()` - Sauvegarde CSV
- `compare_predictions()` - Compare les modèles

### 10. **main_pipeline.ipynb** ⭐
**NOTEBOOK À EXÉCUTER** - Orchestre tous les modules.

```
ÉTAPE 1: Chargement et Exploration
ÉTAPE 2: Préparation des Données
ÉTAPE 3: Ingénierie des Features
ÉTAPE 4: Analyse Exploratoire
ÉTAPE 5: Préparation pour Modélisation
ÉTAPE 6: Entraînement Modèles de Base
ÉTAPE 7: Cross-Validation
ÉTAPE 8: Prédictions Finales
ÉTAPE 9: Résumé Exécutif
```

## 🚀 Utilisation

### Option 1: Exécuter le Pipeline Complet

```bash
# Ouvrir main_pipeline.ipynb dans Jupyter
jupyter notebook main_pipeline.ipynb

# Puis exécuter les cellules dans l'ordre
```

### Option 2: Utiliser les Modules Individuellement

```python
# Exemple: accéder uniquement au data loading
from data_loading import load_data, explore_data
X_train, y_train, X_test, sample_submission = load_data()
explore_data(X_train, y_train, X_test)

# Exemple: créer les features
from feature_engineering import create_all_features
train_data, X_test, all_features = create_all_features(train_data, X_test)

# Exemple: entraîner LightGBM seulement
from model_training import train_lightgbm
model, train_acc, val_acc = train_lightgbm(X_train, y_train, X_val, y_val)
```

### Option 3: Personnaliser la Configuration

```python
# config.py
# Modifier les hyperparamètres:
LGBM_PARAMS = {
    "learning_rate": 0.1,  # Plus rapide
    "max_depth": 7,         # Plus profond
    ...
}
```

## 📊 Fichiers de Sortie

Les fichiers de soumission générés:
- `submission_lgbm_final.csv` - Modèle LightGBM entraîné sur toutes les données
- `submission_ensemble.csv` - Moyenne des prédictions des 8 folds CV (⭐ **Recommandé**)
- `submission_ridge_baseline.csv` - Baseline Ridge

## 🔄 Workflow de Développement

### Pour tester une nouvelle idée:

1. **Feature Engineering**: Ajouter dans `feature_engineering.py`
```python
# Dans create_all_features()
train_data['NEW_FEATURE'] = ...
X_test['NEW_FEATURE'] = ...
```

2. **Modifier config**: Ajouter le nouveau paramètre si nécessaire
```python
# config.py
LGBM_PARAMS['new_param'] = value
```

3. **Tester dans main_pipeline.ipynb**: Les modules chargeront automatiquement

### Pour tester un nouveau modèle:

1. **Créer la fonction** dans `model_training.py`
```python
def train_my_model(X_train, y_train, X_val, y_val):
    model = MyModel()
    model.fit(X_train, y_train)
    return model, train_acc, val_acc
```

2. **Ajouter à `train_base_models()`**
```python
my_model, mt_train, mt_val = train_my_model(...)
results['MyModel'] = {...}
```

3. **Exécuter main_pipeline.ipynb**

## ✨ Avantages de cette Structure

✅ **Modulaire**: Chaque fonction a une responsabilité unique  
✅ **Réutilisable**: Les modules peuvent être importés n'importe où  
✅ **Testable**: Facile d'écrire des tests unitaires  
✅ **Maintenable**: Code lisible et organisé  
✅ **Évolutif**: Facile d'ajouter de nouvelles features/modèles  
✅ **Configurable**: Tous les paramètres en un seul fichier  

## 📝 Notes

- Tous les modules utilisent des chemins **relatifs**
- Les fichiers CSV sont chargés du répertoire courant
- La structure respecte la **temporalité** des données (important!)
- La métrique utilisée est l'**Accuracy** (prédiction du signe)

## 🔗 Prochaines Étapes

- Implémenter l'optimisation des hyperparamètres (Optuna)
- Ajouter des techniques d'ensemble (stacking, blending)
- Implémenter une feature selection automatique
- Ajouter des tests unitaires
- Créer une API Flask pour les prédictions

---

**Version**: 1.0  
**Date**: Janvier 2026  
**Statut**: Production-Ready ✅
