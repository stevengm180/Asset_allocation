"""
Fonctions utilitaires et helpers
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# MÉTRIQUES
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calcule plusieurs métriques de classification.
    
    Parameters
    ----------
    y_true : array-like
        Labels vrais (0 ou 1)
    y_pred : array-like
        Prédictions (0 ou 1)
    y_pred_proba : array-like, optional
        Probabilités prédites
        
    Returns
    -------
    dict
        Dictionnaire contenant les métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc_roc'] = np.nan
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Affiche les métriques de manière formatée.
    """
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
    if 'auc_roc' in metrics and not np.isnan(metrics['auc_roc']):
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_model_comparison(results_df):
    """
    Plot pour comparer les performances des modèles.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['Train Accuracy'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Val Accuracy'], width, label='Val', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparaison des modèles de base')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(feature_importance_df, top_n=20):
    """
    Plot pour afficher l'importance des features.
    """
    top_features = feature_importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features['Importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Top {top_n} Features par Importance')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax


def plot_target_distribution(y):
    """
    Plot pour afficher la distribution de la cible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Histogram
    axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Rendement futur')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution des rendements futurs')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Bar plot classe
    positive = (y > 0).sum()
    negative = (y <= 0).sum()
    axes[1].bar(['Positif', 'Négatif'], [positive, negative])
    axes[1].set_ylabel('Nombre d\'observations')
    axes[1].set_title('Balance des classes')
    
    plt.tight_layout()
    return fig, axes


def plot_correlation_analysis(train_data, all_features, top_n=20):
    """
    Plot pour afficher les corrélations avec la cible.
    """
    correlations = train_data[all_features + ['target']].corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Top corrélations
    top_corr = train_data[correlations.head(15).index.tolist() + ['target']].corr()['target'].drop('target').sort_values()
    top_corr.plot(kind='barh', ax=axes[0], color=['red' if x < 0 else 'green' for x in top_corr])
    axes[0].set_xlabel('Corrélation avec la cible')
    axes[0].set_title('Top 15 features corrélées avec la cible')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    
    # Distribution des corrélations
    axes[1].hist(correlations.values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Corrélation absolue')
    axes[1].set_ylabel('Nombre de features')
    axes[1].set_title('Distribution des corrélations avec la cible')
    axes[1].axvline(x=correlations.mean(), color='red', linestyle='--', label=f'Moyenne: {correlations.mean():.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    return fig, axes, correlations


def plot_cv_scores(cv_scores, mean_score, std_score):
    """
    Plot pour afficher les scores de cross-validation.
    """
    n_splits = len(cv_scores)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, n_splits+1), cv_scores, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.4f}')
    ax.fill_between(range(1, n_splits+1), mean_score - std_score, mean_score + std_score, alpha=0.2)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Evolution de l\'Accuracy par fold (CV LightGBM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# AFFICHAGE DE RAPPORTS
# ============================================================================

def print_summary_report(X, y, X_test, all_features, mean_score, std_score, 
                        feature_importance_df, summary_df, submissions_info):
    """
    Affiche un résumé complet de l'analyse.
    """
    print("=" * 70)
    print("RÉSUMÉ EXÉCUTIF - PRÉDICTION ALLOCATIONS D'ACTIFS")
    print("=" * 70)

    print(f"\n📊 DONNÉES:")
    print(f"  - Ensemble d'entraînement: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  - Ensemble de test: {X_test.shape[0]:,} samples")
    print(f"  - Équilibre des classes: {y.mean():.1%} positif / {(1-y.mean()):.1%} négatif")

    print(f"\n🔧 FEATURES ENGINEERING:")
    print(f"  - Features de base: 41")
    print(f"  - Features dérivées: {len(all_features) - 41}")
    print(f"  - Total features: {len(all_features)}")

    print(f"\n🤖 MODÈLES (VALIDATION SPLIT):")
    for _, row in summary_df.iterrows():
        print(f"  - {row['Model']}: {row['Val Accuracy']:.4f}")

    print(f"\n✓ CROSS-VALIDATION LIGHTGBM ({len(mean_score)} folds):")
    print(f"  - Mean Accuracy: {np.mean(mean_score):.4f}")
    print(f"  - Std Dev: {np.std(mean_score):.4f}")

    print(f"\n🎯 TOP 5 FEATURES:")
    for i, (feat, imp) in enumerate(feature_importance_df.head(5).values, 1):
        print(f"  {i}. {feat}: {imp:.2f}")

    print(f"\n📈 PRÉDICTIONS GÉNÉRÉES:")
    for name, info in submissions_info.items():
        print(f"  - {name}")
        print(f"    → {info['positive']:,} positif ({info['percentage']:.2%})")

    print(f"\n💾 FICHIERS SAUVEGARDÉS:")
    for name in submissions_info.keys():
        print(f"  ✓ {name}.csv")

    print("\n" + "=" * 70)
    print("RECOMMANDATION: Soumettre 'submission_ensemble.csv' (moyenne CV)")
    print("=" * 70)
