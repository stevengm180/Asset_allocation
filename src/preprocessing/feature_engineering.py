"""
Module d'ingénierie des features
"""

import pandas as pd
import numpy as np
from config import RET_COLUMNS, SIGNED_VOLUME_COLUMNS, FEATURE_HORIZONS


def create_all_features(train_data, X_test):
    """
    Crée toutes les features additionnelles.
    
    Parameters
    ----------
    train_data : pd.DataFrame
    X_test : pd.DataFrame
        
    Returns
    -------
    tuple
        (train_data_enriched, X_test_enriched, all_features_list)
    """
    print("📈 Construction des features additionnelles...\n")
    
    # === GROUPE 1: Moyennes des rendements ===
    for horizon in FEATURE_HORIZONS:
        train_data[f'AVERAGE_PERF_{horizon}'] = train_data[RET_COLUMNS[:horizon]].mean(1)
        X_test[f'AVERAGE_PERF_{horizon}'] = X_test[RET_COLUMNS[:horizon]].mean(1)
    print("✓ Moyennes des rendements")

    # === GROUPE 2: Moyennes agrégées par date ===
    for horizon in FEATURE_HORIZONS:
        train_data[f'ALLOCATIONS_AVERAGE_PERF_{horizon}'] = train_data.groupby('TS')[f'AVERAGE_PERF_{horizon}'].transform('mean')
        X_test[f'ALLOCATIONS_AVERAGE_PERF_{horizon}'] = X_test.groupby('TS')[f'AVERAGE_PERF_{horizon}'].transform('mean')
    print("✓ Moyennes agrégées par date")

    # === GROUPE 3: Volatilité ===
    train_data['STD_PERF_20'] = train_data[RET_COLUMNS].std(1)
    train_data['ALLOCATIONS_STD_PERF_20'] = train_data.groupby('TS')['STD_PERF_20'].transform('mean')
    X_test['STD_PERF_20'] = X_test[RET_COLUMNS].std(1)
    X_test['ALLOCATIONS_STD_PERF_20'] = X_test.groupby('TS')['STD_PERF_20'].transform('mean')
    print("✓ Volatilité")

    # === GROUPE 4: Statistiques additionnelles ===
    train_data['SKEW_PERF_20'] = train_data[RET_COLUMNS].skew(1)
    train_data['KURT_PERF_20'] = train_data[RET_COLUMNS].kurtosis(1)
    train_data['MIN_PERF_20'] = train_data[RET_COLUMNS].min(1)
    train_data['MAX_PERF_20'] = train_data[RET_COLUMNS].max(1)
    train_data['RANGE_PERF_20'] = train_data['MAX_PERF_20'] - train_data['MIN_PERF_20']
    
    X_test['SKEW_PERF_20'] = X_test[RET_COLUMNS].skew(1)
    X_test['KURT_PERF_20'] = X_test[RET_COLUMNS].kurtosis(1)
    X_test['MIN_PERF_20'] = X_test[RET_COLUMNS].min(1)
    X_test['MAX_PERF_20'] = X_test[RET_COLUMNS].max(1)
    X_test['RANGE_PERF_20'] = X_test['MAX_PERF_20'] - X_test['MIN_PERF_20']
    print("✓ Statistiques additionnelles")

    # === GROUPE 5: Momentum ===
    train_data['RECENT_PERF_5'] = train_data[RET_COLUMNS[:5]].mean(1)
    train_data['EARLY_PERF_5'] = train_data[RET_COLUMNS[-5:]].mean(1)
    train_data['MOMENTUM_RATIO'] = train_data['RECENT_PERF_5'] / (train_data['EARLY_PERF_5'] + 1e-8)
    
    X_test['RECENT_PERF_5'] = X_test[RET_COLUMNS[:5]].mean(1)
    X_test['EARLY_PERF_5'] = X_test[RET_COLUMNS[-5:]].mean(1)
    X_test['MOMENTUM_RATIO'] = X_test['RECENT_PERF_5'] / (X_test['EARLY_PERF_5'] + 1e-8)
    print("✓ Features de momentum")

    # === GROUPE 6: Volume ===
    train_data['AVERAGE_VOLUME_20'] = train_data[SIGNED_VOLUME_COLUMNS].mean(1)
    train_data['STD_VOLUME_20'] = train_data[SIGNED_VOLUME_COLUMNS].std(1)
    train_data['RECENT_VOLUME_5'] = train_data[SIGNED_VOLUME_COLUMNS[:5]].mean(1)
    
    X_test['AVERAGE_VOLUME_20'] = X_test[SIGNED_VOLUME_COLUMNS].mean(1)
    X_test['STD_VOLUME_20'] = X_test[SIGNED_VOLUME_COLUMNS].std(1)
    X_test['RECENT_VOLUME_5'] = X_test[SIGNED_VOLUME_COLUMNS[:5]].mean(1)
    print("✓ Features de volume")

    # === GROUPE 7: Interactions ===
    train_data['TURNOVER_PERF_INTERACTION'] = train_data['MEDIAN_DAILY_TURNOVER'] * train_data['AVERAGE_PERF_20']
    X_test['TURNOVER_PERF_INTERACTION'] = X_test['MEDIAN_DAILY_TURNOVER'] * X_test['AVERAGE_PERF_20']
    print("✓ Interactions")

    # === GROUPE 8: Agrégations par groupe ===
    train_data['GROUP_AVERAGE_PERF'] = train_data.groupby('GROUP')['AVERAGE_PERF_20'].transform('mean')
    train_data['GROUP_AVERAGE_TURNOVER'] = train_data.groupby('GROUP')['MEDIAN_DAILY_TURNOVER'].transform('mean')
    X_test['GROUP_AVERAGE_PERF'] = X_test.groupby('GROUP')['AVERAGE_PERF_20'].transform('mean')
    X_test['GROUP_AVERAGE_TURNOVER'] = X_test.groupby('GROUP')['MEDIAN_DAILY_TURNOVER'].transform('mean')
    print("✓ Agrégations par groupe")

    # === GROUPE 9: Features avancées (non-linéaires) ===
    # Ratios et interactions
    train_data['PERF_VOLUME_RATIO'] = train_data['AVERAGE_PERF_20'] / (train_data['AVERAGE_VOLUME_20'] + 1e-8)
    train_data['PERF_VOLATILITY_RATIO'] = train_data['AVERAGE_PERF_20'] / (train_data['STD_PERF_20'] + 1e-8)
    train_data['SHARPE_PROXY'] = train_data['AVERAGE_PERF_20'] / (train_data['STD_PERF_20'] + 1e-8)
    
    X_test['PERF_VOLUME_RATIO'] = X_test['AVERAGE_PERF_20'] / (X_test['AVERAGE_VOLUME_20'] + 1e-8)
    X_test['PERF_VOLATILITY_RATIO'] = X_test['AVERAGE_PERF_20'] / (X_test['STD_PERF_20'] + 1e-8)
    X_test['SHARPE_PROXY'] = X_test['AVERAGE_PERF_20'] / (X_test['STD_PERF_20'] + 1e-8)
    
    # Tendances
    train_data['PERF_TREND_RECENT_vs_EARLY'] = train_data['RECENT_PERF_5'] - train_data['EARLY_PERF_5']
    train_data['VOLUME_TREND'] = train_data['RECENT_VOLUME_5'] - train_data[SIGNED_VOLUME_COLUMNS[-5:]].mean(1)
    
    X_test['PERF_TREND_RECENT_vs_EARLY'] = X_test['RECENT_PERF_5'] - X_test['EARLY_PERF_5']
    X_test['VOLUME_TREND'] = X_test['RECENT_VOLUME_5'] - X_test[SIGNED_VOLUME_COLUMNS[-5:]].mean(1)
    
    # Transformations non-linéaires
    train_data['PERF_ABS'] = train_data['AVERAGE_PERF_20'].abs()
    train_data['PERF_SQUARED'] = train_data['AVERAGE_PERF_20'] ** 2
    train_data['PERF_LOG'] = np.sign(train_data['AVERAGE_PERF_20']) * np.log1p(train_data['AVERAGE_PERF_20'].abs())
    
    X_test['PERF_ABS'] = X_test['AVERAGE_PERF_20'].abs()
    X_test['PERF_SQUARED'] = X_test['AVERAGE_PERF_20'] ** 2
    X_test['PERF_LOG'] = np.sign(X_test['AVERAGE_PERF_20']) * np.log1p(X_test['AVERAGE_PERF_20'].abs())
    
    # Z-score normalisé par groupe
    train_data['PERF_ZSCORE_GROUP'] = train_data.groupby('GROUP')['AVERAGE_PERF_20'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    X_test['PERF_ZSCORE_GROUP'] = X_test.groupby('GROUP')['AVERAGE_PERF_20'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # Percentile ranking par groupe
    train_data['PERF_PERCENTILE_GROUP'] = train_data.groupby('GROUP')['AVERAGE_PERF_20'].transform(
        lambda x: x.rank(pct=True)
    )
    X_test['PERF_PERCENTILE_GROUP'] = X_test.groupby('GROUP')['AVERAGE_PERF_20'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Multi-horizon momentum
    train_data['MOMENTUM_3_5'] = train_data['AVERAGE_PERF_3'] - train_data['AVERAGE_PERF_5']
    train_data['MOMENTUM_5_10'] = train_data['AVERAGE_PERF_5'] - train_data['AVERAGE_PERF_10']
    train_data['MOMENTUM_10_20'] = train_data['AVERAGE_PERF_10'] - train_data['AVERAGE_PERF_20']
    
    X_test['MOMENTUM_3_5'] = X_test['AVERAGE_PERF_3'] - X_test['AVERAGE_PERF_5']
    X_test['MOMENTUM_5_10'] = X_test['AVERAGE_PERF_5'] - X_test['AVERAGE_PERF_10']
    X_test['MOMENTUM_10_20'] = X_test['AVERAGE_PERF_10'] - X_test['AVERAGE_PERF_20']
    
    # Persistance de la performance
    train_data['PERF_PERSISTENCE'] = train_data['RECENT_PERF_5'] * train_data['AVERAGE_PERF_20']
    X_test['PERF_PERSISTENCE'] = X_test['RECENT_PERF_5'] * X_test['AVERAGE_PERF_20']
    
    # Stabilité (inverse de la volatilité)
    train_data['STABILITY'] = 1.0 / (train_data['STD_PERF_20'] + 1e-8)
    X_test['STABILITY'] = 1.0 / (X_test['STD_PERF_20'] + 1e-8)
    
    # Drawdown approximatif
    train_data['POTENTIAL_DRAWDOWN'] = train_data['MIN_PERF_20'] - train_data['AVERAGE_PERF_20']
    X_test['POTENTIAL_DRAWDOWN'] = X_test['MIN_PERF_20'] - X_test['AVERAGE_PERF_20']
    
    print("✓ Features avancées (ratio, tendances, interactions)")

    # Construction de la liste complète
    all_features = RET_COLUMNS + SIGNED_VOLUME_COLUMNS + ['MEDIAN_DAILY_TURNOVER']
    all_features += [f'AVERAGE_PERF_{i}' for i in FEATURE_HORIZONS]
    all_features += [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in FEATURE_HORIZONS]
    all_features += ['STD_PERF_20', 'ALLOCATIONS_STD_PERF_20']
    all_features += ['SKEW_PERF_20', 'KURT_PERF_20', 'MIN_PERF_20', 'MAX_PERF_20', 'RANGE_PERF_20']
    all_features += ['RECENT_PERF_5', 'EARLY_PERF_5', 'MOMENTUM_RATIO']
    all_features += ['AVERAGE_VOLUME_20', 'STD_VOLUME_20', 'RECENT_VOLUME_5']
    all_features += ['TURNOVER_PERF_INTERACTION']
    all_features += ['GROUP_AVERAGE_PERF', 'GROUP_AVERAGE_TURNOVER']
    # Features avancées
    all_features += ['PERF_VOLUME_RATIO', 'PERF_VOLATILITY_RATIO', 'SHARPE_PROXY']
    all_features += ['PERF_TREND_RECENT_vs_EARLY', 'VOLUME_TREND']
    all_features += ['PERF_ABS', 'PERF_SQUARED', 'PERF_LOG']
    all_features += ['PERF_ZSCORE_GROUP', 'PERF_PERCENTILE_GROUP']
    all_features += ['MOMENTUM_3_5', 'MOMENTUM_5_10', 'MOMENTUM_10_20']
    all_features += ['PERF_PERSISTENCE', 'STABILITY', 'POTENTIAL_DRAWDOWN']

    print(f"\n✓ Feature engineering AVANCÉ complet!")
    print(f"  Total features: {len(all_features)}")
    print(f"    - Base: 41")
    print(f"    - Dérivées: {len(all_features) - 41}")
    
    return train_data, X_test, all_features
