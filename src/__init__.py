# Exemple de structure pour config.py en tant que module

"""
Configuration centralisée pour le projet
Utilisé par tous les modules
"""

import os
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Chemins de données
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_PATH = PROJECT_ROOT / "data" / "external"

# Chemins de modèles et outputs
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

# Créer les répertoires s'ils n'existent pas
for path in [DATA_PROCESSED_PATH, DATA_EXTERNAL_PATH, MODELS_PATH / "saved_models", OUTPUTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Configuration des modèles
class ModelConfig:
    """Configuration par défaut pour l'entraînement des modèles"""
    test_size = 0.2
    random_state = 42
    n_splits = 5
    
class PreprocessingConfig:
    """Configuration du prétraitement"""
    handle_missing = True
    scale_features = True
    remove_outliers = False
