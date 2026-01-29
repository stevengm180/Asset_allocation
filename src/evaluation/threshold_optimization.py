"""
Module d'optimisation des seuils de prédiction
Trouve le seuil optimal pour chaque modèle
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def find_optimal_threshold(y_true, y_pred_proba, step=0.01):
    """
    Trouve le seuil qui maximise l'accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray ou pd.Series
        Labels vrais (0 ou 1)
    y_pred_proba : np.ndarray
        Probabilités prédites (entre 0 et 1)
    step : float
        Taille du pas pour tester les seuils
        
    Returns
    -------
    dict
        {'threshold': float, 'accuracy': float, 'metrics': dict}
    """
    print("🎯 OPTIMISATION DU SEUIL DE PRÉDICTION\n")
    
    best_threshold = 0.5
    best_accuracy = 0
    best_metrics = {}
    results = []
    
    for threshold in np.arange(0.3, 0.7, step):
        y_pred = (y_pred_proba > threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    print(f"✨ MEILLEUR SEUIL: {best_threshold:.3f}")
    print(f"\n   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1']:.4f}")
    
    return {
        'threshold': best_threshold,
        'accuracy': best_accuracy,
        'metrics': best_metrics,
        'all_results': results
    }


def apply_optimized_threshold(y_pred_proba, optimal_threshold):
    """
    Applique le seuil optimal aux prédictions.
    
    Parameters
    ----------
    y_pred_proba : np.ndarray
        Probabilités prédites
    optimal_threshold : float
        Seuil optimal
        
    Returns
    -------
    np.ndarray
        Prédictions binaires avec le seuil optimal
    """
    return (y_pred_proba > optimal_threshold).astype(int)


def find_threshold_for_balance(y_true, y_pred_proba, target_positive_ratio=None):
    """
    Trouve le seuil qui atteint un ratio de positifs cible.
    
    Parameters
    ----------
    y_true : np.ndarray
    y_pred_proba : np.ndarray
    target_positive_ratio : float, optional
        Ratio cible de positifs. Si None, utilise le ratio de y_true.
        
    Returns
    -------
    float
        Seuil optimal
    """
    if target_positive_ratio is None:
        target_positive_ratio = (y_true > 0).mean()
    
    print(f"🔄 OPTIMISATION POUR RATIO ÉQUILIBRÉ\n")
    print(f"   Ratio cible de positifs: {target_positive_ratio:.2%}")
    
    # Trier les probabilités
    sorted_indices = np.argsort(y_pred_proba)
    
    # Trouver le seuil qui atteint le ratio cible
    n_positive = int(len(y_pred_proba) * target_positive_ratio)
    threshold = y_pred_proba[sorted_indices[-(n_positive+1)]]
    
    print(f"   Seuil trouvé: {threshold:.3f}")
    
    y_pred = (y_pred_proba > threshold).astype(int)
    actual_ratio = y_pred.mean()
    
    print(f"   Ratio atteint: {actual_ratio:.2%}")
    
    return threshold


def plot_threshold_optimization(results):
    """
    Affiche les courbes d'optimisation des seuils.
    """
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy par Seuil')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(results_df['threshold'], results_df['precision'], 'o-', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision par Seuil')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results_df['threshold'], results_df['recall'], 'o-', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall par Seuil')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results_df['threshold'], results_df['f1'], 'o-', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score par Seuil')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def compare_thresholds(y_true, y_pred_proba, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    Compare plusieurs seuils.
    
    Parameters
    ----------
    y_true : np.ndarray
    y_pred_proba : np.ndarray
    thresholds : list
        Seuils à comparer
        
    Returns
    -------
    pd.DataFrame
        Comparaison des métriques
    """
    import pandas as pd
    
    print("📊 COMPARAISON DE PLUSIEURS SEUILS\n")
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            '# Positifs': y_pred.sum()
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


# Import pandas pour plot
import pandas as pd
