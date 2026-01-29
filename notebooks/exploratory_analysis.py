"""
Module d'analyse exploratoire des features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_correlation_analysis


def analyze_correlations(train_data, all_features):
    """
    Analyse les corrélations des features avec la cible.
    
    Parameters
    ----------
    train_data : pd.DataFrame
    all_features : list
        
    Returns
    -------
    pd.Series
        Corrélations absolues triées
    """
    print("📊 Analyse des corrélations avec la cible:\n")
    
    correlations = train_data[all_features + ['target']].corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    print("Top 20 features les plus corrélées:")
    print(correlations.head(20).to_string())
    
    print(f"\nCorrélation moyenne avec la cible: {correlations.mean():.4f}")
    print(f"Max corrélation: {correlations.max():.4f}")
    print(f"Min corrélation: {correlations.min():.4f}")
    
    return correlations


def plot_correlations(train_data, all_features):
    """
    Crée les plots de corrélation.
    """
    fig, axes, correlations = plot_correlation_analysis(train_data, all_features, top_n=20)
    plt.show()
    return correlations


def analyze_features_by_class(train_data, all_features, correlations):
    """
    Analyse les features par classe (positif vs négatif).
    
    Parameters
    ----------
    train_data : pd.DataFrame
    all_features : list
    correlations : pd.Series
        
    Returns
    -------
    pd.DataFrame
        Comparaison des moyennes par classe
    """
    print("\n📈 Analyse des features par classe:\n")
    
    positive_mask = train_data['target'] > 0
    negative_mask = train_data['target'] <= 0
    
    top_20_features = correlations.head(20).index.tolist()
    
    comparison_df = pd.DataFrame({
        'Feature': top_20_features,
        'Mean_Positive': [train_data.loc[positive_mask, feat].mean() for feat in top_20_features],
        'Mean_Negative': [train_data.loc[negative_mask, feat].mean() for feat in top_20_features],
    })
    
    comparison_df['Difference'] = comparison_df['Mean_Positive'] - comparison_df['Mean_Negative']
    
    print(comparison_df.to_string())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    comparison_df.sort_values('Difference', ascending=True).plot(
        x='Feature', 
        y=['Mean_Positive', 'Mean_Negative'], 
        kind='barh',
        ax=ax
    )
    ax.set_xlabel('Moyenne de la feature')
    ax.set_title('Moyennes des top features par classe')
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def get_feature_statistics(train_data, all_features):
    """
    Fournit des statistiques complètes sur les features.
    
    Returns
    -------
    dict
        Statistiques pour chaque feature
    """
    stats = {}
    
    for feat in all_features:
        stats[feat] = {
            'mean': train_data[feat].mean(),
            'std': train_data[feat].std(),
            'min': train_data[feat].min(),
            'max': train_data[feat].max(),
            'missing': train_data[feat].isnull().sum(),
        }
    
    return stats
