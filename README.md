# OptInvest 

## Description

OptInvest est une application d'analyse et de simulation de stratégies d'investissement sur les marchés financiers. Elle permet de comparer différentes approches d'investissement progressif (Dollar-Cost Averaging) et d'investissement en une fois (Lump Sum) sur des données historiques réelles, puis de générer des prédictions basées sur des modèles statistiques.

## Installation

1. Clone le projet depuis GitHub

2. Installe uv

```bash
pip install uv
```

3. Crée et active l'environnement virtuel

```bash
uv venv
```

```bash
source .venv/bin/activate
```

4. Installe les dépendances dans l'environnement virtuel

```bash
uv add -r requirements.txt
```

NB : vous pouvez aussi utiliser `uv pip install -r requirements.txt` si `uv add` provoque une erreur

5. Lance l'API backend

```bash
python -m app.api.main
```

6. Lance l'interface frontend

Dans un autre terminal

```bash	
streamlit run app/client/main.py
```

7. Accède à l'application dans le navigateur : `http://localhost:8501`

## Fonctionnalités métier

### Simulation de portefeuille
- Analyse historique de portefeuilles sur des actifs réels (actions, ETF)
- Support de différentes fréquences d'investissement (mensuelle, trimestrielle, semestrielle, annuelle)
- Prise en compte des frais de gestion annuels
- Calcul de métriques de performance (CAGR, rendement total)
- NB : Le choix des actifs est limité à 6 actifs réels pour l'instant


### Comparaison de stratégies d'investissement
- Dollar-Cost Averaging (DCA) à différentes fréquences
- Lump Sum (investissement en une fois)
- Visualisation comparative des performances historiques
- Benchmarking avec l'indice mondial ACWI IMI

### Modélisation prédictive
- Prédiction de l'évolution future des portefeuilles par régression linéaire
- Analyse de la qualité des modèles prédictifs
- Visualisation de la distribution des résidus
- Exportation des résultats en PDF et Excel

## Stack technique

### Backend
- **FastAPI**: Framework API REST haute performance
- **Python 3.11+**: Langage de programmation principal
- **pandas/numpy**: Manipulation et analyse de données
- **scikit-learn**: Modélisation et prédiction (régression linéaire)
- **yfinance**: Extraction de données financières depuis Yahoo Finance

### Frontend
- **Streamlit**: Interface utilisateur interactive
- **Plotly**: Visualisations graphiques interactives

### Export et rapports
- **ReportLab**: Génération de rapports PDF
- **XlsxWriter**: Export des données en Excel
- **kaleido**: Conversion des graphiques pour les rapports

## Architecture de l'application

### Modules principaux
- **core/**: Logique métier et calculs
  - **simulator.py**: Simulations d'investissement
  - **predictor.py**: Modélisation prédictive
  - **datasources/**: Extraction et gestion des données

- **api/**: Endpoints REST
  - **routers/**: Définition des routes API
  - **schemas/**: Validation des données avec Pydantic

- **client/**: Interface utilisateur
  - **main.py**: Point d'entrée de l'interface
  - **plots_manager.py**: Génération des visualisations
  - **export_manager.py**: Export des rapports
  - **ui_manager.py**: Gestion de l'interface utilisateur

### Endpoints API
- **GET /health**: Vérification de l'état de l'API
- **POST /api/simuler**: Simulation d'un portefeuille d'investissement
- **POST /api/predire**: Prédiction et comparaison des stratégies d'investissement

## Métriques et indicateurs

### Métriques de performance
- **CAGR** (Compound Annual Growth Rate): Taux de croissance annuel composé
- **Rendement total**: Gain net en euros
- **Volatilité annualisée**: Mesure de la dispersion des rendements
- **Ratio de Sharpe**: Rendement ajusté au risque

### Métriques de qualité des prédictions
- **R²**: Coefficient de détermination (qualité globale du modèle)
- **RMSE** (Root Mean Square Error): Erreur quadratique moyenne
- **MAE** (Mean Absolute Error): Erreur absolue moyenne
- **Écart-type des résidus**: Dispersion des erreurs de prédiction



## Points d'amélioration

### Bug
- Bug d'exportation des données qui fait disparaitre le dashboard sur l'UI, il faudra re simuler pour le revoir

### Points Metiers

- Support de plus d'actifs financiers

### Tests statistiques
- Implémenter des tests de normalité plus robustes (actuellement basé sur visualisation)
- Ajouter des tests d'autocorrélation des résidus pour valider les hypothèses de régression


### Packaging et déploiement
- Containerisation avec Docker:
  ```
  # Exemple de Dockerfile à créer
  FROM python:3.11-slim
  
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  
  COPY . .
  
  EXPOSE 8000 8501
  
  CMD ["sh", "-c", "python -m app.api.main & streamlit run app/client/main.py"]
  ```

