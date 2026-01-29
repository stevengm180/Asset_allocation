"""
Configuration centralisée pour le projet de prédiction d'allocations d'actifs
"""

# ============================================================================
# CONFIGURATION DES DONNÉES
# ============================================================================

# Fichiers de données
DATA_FILES = {
    'X_train': 'X_train_9xQjqvZ.csv',
    'y_train': 'y_train_Ppwhaz8.csv',
    'X_test': 'X_test_1zTtEnD.csv',
    'sample_submission': 'sample_submission_SpGVFuH.csv',
}

# Colonnes de base
RET_COLUMNS = [f'RET_{i}' for i in range(1, 21)]
SIGNED_VOLUME_COLUMNS = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
BASE_COLUMNS = RET_COLUMNS + SIGNED_VOLUME_COLUMNS + ['MEDIAN_DAILY_TURNOVER']

# ============================================================================
# CONFIGURATION DES FEATURES
# ============================================================================

FEATURE_HORIZONS = [3, 5, 10, 15, 20]

# ============================================================================
# CONFIGURATION DES MODÈLES
# ============================================================================

# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    # Force single-thread to avoid multiprocessing ResourceTracker errors on exit.
    "num_threads": 1,
    "seed": 42,
    "verbosity": -1,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

LGBM_BOOSTING_ROUNDS = 300

# Ridge Regression
RIDGE_ALPHA = 1.0

# Logistic Regression
LOGISTIC_MAX_ITER = 1000

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# ============================================================================
# CONFIGURATION DE LA CROSS-VALIDATION
# ============================================================================

N_SPLITS_CV = 8
CV_RANDOM_STATE = 0
CV_SHUFFLE = True
TRAIN_TEST_SPLIT_RATIO = 0.7

# ============================================================================
# CONFIGURATION DE L'ANALYSE
# ============================================================================

TOP_FEATURES_DISPLAY = 20
TOP_FEATURES_FOR_SELECTION = 30

# ============================================================================
# CONFIGURATION DE SOUMISSION
# ============================================================================

SUBMISSION_FILES = {
    'lgbm_final': 'submission_lgbm_final.csv',
    'ensemble': 'submission_ensemble.csv',
    'ridge_baseline': 'submission_ridge_baseline.csv',
}
