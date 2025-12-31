# OptInvest 

## Description

OptInvest est une application d'analyse et de simulation de stratégies d'investissement sur les marchés financiers. Elle permet de comparer différentes approches d'investissement progressif (Dollar-Cost Averaging) et en une fois (Lump Sum), d'optimiser les allocations de portefeuille selon la théorie moderne de portefeuille (Markowitz), et de réaliser des prédictions statistiques.

## Installation

### Option 1 : Installation avec Docker (Recommandé)

1. **Installez Docker**
   - Téléchargez et installez Docker Desktop depuis [docker.com](https://www.docker.com/get-started)
   - Vérifiez l'installation : `docker --version`

2. **Clonez le projet depuis GitHub**

3. **Lancez l'application avec Docker Compose**

```bash
docker-compose up
```

4. **Accédez à l'application dans le navigateur : `http://localhost:8501`**

### Option 2 : Installation manuelle avec uv

1. **Clonez le projet depuis GitHub**

2. **Installez Docker** (requis pour certaines fonctionnalités)
   - Téléchargez et installez Docker Desktop depuis [docker.com](https://www.docker.com/get-started)

3. **Installez uv**

```bash
pip install uv
```

4. **Créez et activez l'environnement virtuel**

```bash
uv venv
```

```bash
source .venv/bin/activate
```

5. **Installez les dépendances dans l'environnement virtuel**

```bash
uv add -r requirements.txt
```

NB : vous pouvez aussi utiliser `uv pip install -r requirements.txt` si `uv add` provoque une erreur

6. **Lancez l'API backend** (dans un premier terminal)

```bash
uv run --active python app/api/main.py
```

7. **Lancez l'interface frontend** (dans un second terminal)

```bash
uv run --active streamlit run app/client/main.py
```

8. **Accédez à l'application dans le navigateur : `http://localhost:8501`**

## Fonctionnalités métier

### Simulation de portefeuille
- Analyse historique de portefeuilles sur des actifs réels (actions, ETF)
- Support de différentes fréquences d'investissement (mensuelle, trimestrielle, semestrielle, annuelle)
- Prise en compte des frais de gestion annuels
- Calcul de métriques de performance (CAGR, rendement total, volatilité annualisée, ratio de Sharpe)
- **NB** : Le choix des actifs est limité à 6 actifs parmi : AAPL, MSFT, AGGH, TLT, VWCE.DE, SXR8.DE


### Comparaison de stratégies d'investissement
- **Dollar-Cost Averaging (DCA)** : Investissement progressif à différentes fréquences
- **Lump Sum** : Investissement en une seule fois au début de la période
- Visualisation comparative des performances historiques
- Calcul du prix moyen pondéré pour les portefeuilles multi-actifs

### Optimisation de portefeuille (Markowitz)
- **Frontière efficiente** : Génération de 50 portefeuilles optimaux en faisant varier l'aversion au risque
- **Portefeuille tangent** : Identification du portefeuille maximisant le ratio de Sharpe
- **Optimisation quadratique** : Résolution par la méthode des points intérieurs (`scipy.optimize.minimize`)
- **Contraintes** :
  - Somme des poids = 1 (contrainte budgétaire)
  - Poids ≥ 0 (pas de vente à découvert)
- **Calcul de la matrice de covariance** : Basé sur les rendements journaliers annualisés (252 jours de bourse)
- **Taux sans risque empirique** : Calculé à partir de l'ETF IEF (obligations américaines 7-10 ans)

### Modélisation prédictive
- **Régression linéaire OLS** : Prédiction de l'évolution future du portefeuille en fonction du montant investi
- **Analyse de la qualité du modèle** :
  - **R²** : Coefficient de détermination
  - **RMSE relatif** : Erreur quadratique moyenne en pourcentage
  - **Écart-type des résidus** : Dispersion des erreurs
  - **Statistique de Durbin-Watson** : Test d'autocorrélation des résidus
- **Analyse des coefficients** :
  - P-values pour tester la significativité statistique
  - Intervalles de confiance à 95%
- **Détection d'overfitting** : Comparaison R² train vs validation (split 80/20)
- **Visualisation** : Distribution des résidus comparée à une loi normale
- **Exportation** : Résultats en PDF et Excel
- **Qualité des prédictions** (basée sur R²) :
  - R² > 0.8 : excellente qualité
  - 0.6 < R² ≤ 0.8 : qualité correcte
  - R² < 0.6 : qualité faible


## Stack technique

### Backend
- **FastAPI** : Framework API REST haute performance
- **Python 3.11+** : Langage de programmation principal
- **pandas/numpy** : Manipulation et analyse de données
- **statsmodels** : Modélisation statistique (régression OLS, tests de Durbin-Watson)
- **scikit-learn** : Métriques de qualité (R², train/test split)
- **scipy** : Optimisation quadratique (méthode des points intérieurs)
- **yfinance** : Extraction de données financières depuis Yahoo Finance

### Frontend
- **Streamlit** : Interface utilisateur interactive
- **Plotly** : Visualisations graphiques interactives

## Architecture de l'application

### Modules principaux
- **core/** : Logique métier et calculs
  - **simulator.py** : Simulations d'investissement DCA et Lump Sum
    - Calcul des métriques (CAGR, volatilité, Sharpe)
    - Optimisation de Markowitz (frontière efficiente, portefeuille tangent)
    - Gestion des frais de gestion annuels
  - **predictor.py** : Modélisation prédictive
    - Classe abstraite `Predictor` pour extensibilité
    - `LinearModelPredictor` : Régression OLS avec analyse complète
    - Classes futures (non implémentées) : `AutoRegressiveModelPredictor`, `ARMA1ModelPredictor`, `ARMAGARCH1ModelPredictor`
  - **datasources/** : Extraction et gestion des données
    - `YahooExtractor` : Récupération des prix et volumes avec cache local (parquet)

- **api/** : Endpoints REST
  - **routers/** : Définition des routes API
    - `health.py` : Health check
    - `simulation.py` : Simulation de portefeuille
    - `prediction.py` : Prédictions futures
  - **schemas/** : Validation des données avec Pydantic
    - `SimulationRequest`, `SimulationResponse`
    - `PredictionRequest`, `PredictionResponse`
    - `MarkowitzOptimalResult`
    - Analyses : `FitQualityAnalysis`, `ResidualAnalysis`, `CoefficientAnalysis`, `OverfittingAnalysis`

- **client/** : Interface utilisateur
  - **main.py** : Point d'entrée de l'interface
  - **plots_manager.py** : Génération des visualisations (Plotly)
  - **export_manager.py** : Export des rapports (PDF, Excel)
  - **ui_manager.py** : Gestion de l'interface utilisateur (sidebar, formulaires)

### Endpoints API
- **GET /health** : Vérification de l'état de l'API
- **POST /api/simuler** : Simulation d'un portefeuille d'investissement
  - Paramètres : actifs, durée, montant initial, apports périodiques, fréquence, frais de gestion, stratégie
  - Retour : métriques de performance, valeur du portefeuille dans le temps, résultats Markowitz
- **POST /api/predire** : Prédiction et analyse statistique
  - Paramètres : données historiques (X, y), durée de prédiction, fréquence, modèle
  - Retour : prédictions futures, analyses de qualité, résidus, coefficients

## Métriques et indicateurs

### Métriques de performance
- **CAGR** (Compound Annual Growth Rate) : Taux de croissance annuel composé
  - Formule : `(Valeur_finale / Cash_investi)^(1/durée_ans) - 1`
- **Rendement total** : Gain net en euros
  - Formule : `Valeur_finale - Cash_investi`
- **Volatilité annualisée** : Mesure du risque du portefeuille
  - Calculée à partir de la matrice de covariance des rendements : `sqrt(w^T * Σ * w)`
  - Σ : matrice de covariance annualisée (rendements journaliers × 252)
- **Ratio de Sharpe** : Rendement ajusté au risque
  - Formule : `(Rendement_portefeuille - Taux_sans_risque) / Volatilité`
  - Taux sans risque : Calculé empiriquement à partir de l'ETF IEF

### Optimisation de Markowitz
- **Rendements moyens annualisés** : Rendements journaliers × 252
- **Matrice de covariance annualisée** : Covariance des rendements journaliers × 252
- **Poids optimaux** : Vecteur de poids maximisant le ratio de Sharpe sur la frontière efficiente
- **Fonction objective** : `-[w^T * μ - (γ/2) * w^T * Σ * w]` où γ varie sur [0.01, 100]

### Métriques de qualité des prédictions
- **R²** (Coefficient de détermination) : Part de la variance expliquée par le modèle
- **RMSE relatif** : Erreur quadratique moyenne en % de la moyenne
- **Écart-type des résidus** : Dispersion des erreurs (avec correction ddof=1)
- **Statistique de Durbin-Watson** : Détection d'autocorrélation des résidus
  - DW ≈ 2 : pas d'autocorrélation
  - DW < 2 : autocorrélation positive
  - DW > 2 : autocorrélation négative
- **P-values des coefficients** : Significativit�� statistique (seuil : 0.05)
- **Intervalles de confiance à 95%** : Plage de valeurs plausibles pour les coefficients
- **Différence R² train/validation** : Indicateur d'overfitting



## Points d'amélioration

### Points métier
- Support de plus d'actifs financiers au-delà des 6 actifs actuels
- Implémentation des modèles prédictifs avancés (ARIMA, ARMA-GARCH)

### Optimisation du portefeuille
- Mieux estimer la matrice de covariance $\Sigma$ des rendements des actifs, la matrice de covariance empirique n'est pas robuste.


## Documentation mathématique

Pour une description détaillée des concepts mathématiques et des formules utilisées, consultez le fichier [CONCEPTS.md](./CONCEPTS.md) qui couvre :
- Stratégies d'investissement (DCA, Lump Sum)
- Métriques de performance (CAGR, volatilité, Sharpe)
- Optimisation de portefeuille (théorie de Markowitz, frontière efficiente)
- Modélisation prédictive (régression OLS, analyse des résidus)
- Métriques de qualité des modèles (R², RMSE, Durbin-Watson, overfitting)

## Aperçu de l'application
- Aperçu global :

<img width="1811" height="965" alt="image" src="https://github.com/user-attachments/assets/932d8b5d-62f3-4555-a986-d398583e36f0" />




- Graphiques :

<img width="1281" height="447" alt="image" src="https://github.com/user-attachments/assets/2f757e73-b3da-4985-90a0-aabd29beb759" />

<img width="1281" height="447" alt="image" src="https://github.com/user-attachments/assets/bd2faf83-3003-4209-8143-bdcaad156be1" />

<img width="1284" height="698" alt="image" src="https://github.com/user-attachments/assets/be64d415-b0aa-4321-b0be-eabe309620c8" />

<img width="1279" height="843" alt="image" src="https://github.com/user-attachments/assets/082da289-11af-4ad5-a0ac-2da3d79fd5d2" />

<img width="1242" height="428" alt="image" src="https://github.com/user-attachments/assets/750ca4cb-b09d-4243-b205-0876d16ae49b" />
