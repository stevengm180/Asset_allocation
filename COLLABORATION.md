# Structure de travail collaboratif

## 📁 Organisation du projet

```
Allocation actifs/
├── src/                          # Code réutilisable
│   ├── preprocessing/            # Pipeline de prétraitement des données
│   │   ├── data_loading.py
│   │   ├── data_preparation.py
│   │   └── feature_engineering.py
│   ├── models/                   # Modèles et entraînement
│   │   ├── model_training.py
│   │   ├── stacking_ensemble.py
│   │   ├── optimization.py
│   │   ├── predictions.py
│   │   └── config.py
│   └── evaluation/               # Évaluation et validation
│       ├── cross_validation.py
│       └── threshold_optimization.py
│
├── notebooks/                    # Exploration et documentation
│   ├── exploratory_analysis.ipynb
│   ├── main_pipeline.ipynb
│   ├── main_optimized_pipeline.ipynb
│   ├── benchmark_submission.ipynb
│   └── research_structure.ipynb
│
├── data/                         # Gestion des données
│   ├── raw/                      # ⚠️ Données brutes (immuable, ne pas modifier)
│   ├── processed/                # Données prétraitées
│   └── external/                 # Données externes
│
├── models/                       # Modèles entraînés
│   └── saved_models/
│
├── outputs/                      # Résultats et prédictions
│   └── predictions/
│
├── submissions/                  # Soumissions finales
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── COMPARISON_MODELS.md
│   └── OPTIMIZATIONS.md
│
├── tests/                        # Tests unitaires
│
├── config/                       # Configuration
│
├── .gitignore
└── requirements.txt              # Dépendances Python
```

## 🚀 Workflow collaboratif

### Pour prétraiter des données :
```python
from src.preprocessing.data_loading import load_data
from src.preprocessing.data_preparation import prepare_data
from src.preprocessing.feature_engineering import engineer_features

data = load_data('data/raw/...')
data = prepare_data(data)
data = engineer_features(data)
```

### Pour entraîner un modèle :
```python
from src.models.model_training import train_model
from src.models.config import ModelConfig

config = ModelConfig()
model = train_model(data, config)
```

### Pour évaluer :
```python
from src.evaluation.cross_validation import cross_validate
from src.models.predictions import predict

cv_results = cross_validate(model, data)
predictions = predict(model, test_data)
```

## 📋 Convention de branchage

- `main` - version stable
- `dev` - développement commun
- `feature/preprocessing-*` - nouvelles méthodes de prétraitement
- `feature/models-*` - nouveaux modèles
- `feature/optimization-*` - optimisations
- `experiment/*` - expériences temporaires

## ⚠️ Règles importantes

1. **Ne jamais modifier** `data/raw/` - c'est le référentiel de données brutes
2. **Données traitées** dans `data/processed/` avec timestamps
3. **Modèles** sauvegardés dans `models/saved_models/`
4. **Résultats** dans `outputs/` avec date et heure
5. **Chaque branche** peut avoir ses propres résultats sans conflit

## 🔄 Avant de push

```bash
# Mettez à jour les dépendances
pip freeze > requirements.txt

# Vérifiez que les chemins sont relatifs à la racine du projet
# Utilisez des imports depuis src/ (ex: from src.preprocessing import ...)
```
