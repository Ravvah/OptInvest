# Concepts mathématiques utilisés dans OptInvest

## 1. Objectif

OptInvest compare deux stratégies d'investissement :  
- **DCA (Dollar-Cost Averaging)** : investir progressivement à intervalles réguliers  
- **Lump Sum** : investir tout le capital en une seule fois au début  

Le projet simule l'évolution du portefeuille selon chaque approche, optimise la répartition des actifs selon la théorie de Markowitz, puis analyse et prédit la performance future.

---

## 2. Stratégies d'investissement

### DCA (investissement progressif)

À chaque période $k$, on investit un montant $c_k$ lorsque le prix moyen du portefeuille est $P_{t_k}$.  
Le nombre de parts achetées est :

$$
s_k = \frac{c_k}{P_{t_k}}
$$

- $s_k$ : nombre de parts achetées à la période $k$
- $c_k$ : montant investi à la période $k$ (montant initial + apports périodiques)
- $P_{t_k}$ : prix moyen du portefeuille à la période $t_k$

Le nombre total de parts évolue au fil du temps :

$$
S_t = \sum_{k=0}^{t} s_k
$$

La valeur du portefeuille au temps $T$ est :

$$
V_T^{\text{DCA}} = S_T \cdot P_T
$$

où $P_T$ est le prix moyen du portefeuille à la date finale.

Les frais de gestion annuels sont appliqués chaque année en réduisant le nombre de parts :

$$
S_{t+1} = S_t \times (1 - f)
$$

où $f$ est le taux de frais de gestion annuel.

---

### Lump Sum (investissement unique)

On investit un montant total $C$ au début, lorsque le prix moyen est $P_{t_0}$.  
La valeur finale est :

$$
V_T^{\text{LS}} = C \frac{P_T}{P_{t_0}} \times \prod_{i=1}^{T} (1 - f)
$$

- $V_T^{\text{LS}}$ : valeur du portefeuille à la date finale $T$
- $C$ : montant total investi au départ
- $P_{t_0}$ : prix moyen du portefeuille au moment de l'investissement
- $f$ : frais de gestion annuels

---

## 3. Mesures de performance

Pour comparer les stratégies, OptInvest calcule :

### Rendement total
Différence entre la valeur finale et le montant total investi :

$$
R_{\text{total}} = V_T - C_{\text{investi}}
$$

### CAGR (taux de croissance annuel composé)

$$
\text{CAGR} = \left( \frac{V_f}{V_i} \right)^{1/T} - 1
$$

- $V_f$ : valeur finale du portefeuille
- $V_i$ : montant total investi
- $T$ : durée en années

### Volatilité annualisée du portefeuille

La volatilité du portefeuille global est calculée à partir de la matrice de covariance des rendements des actifs individuels :

$$
\sigma_{\text{portefeuille}} = \sqrt{w^T \cdot \Sigma \cdot w}
$$

où :
- $w$ : vecteur des poids des actifs dans le portefeuille (somme = 1)
- $\Sigma$ : matrice de covariance annualisée des rendements des actifs
- $w^T$ : transposée du vecteur des poids

La matrice de covariance $\Sigma$ est calculée à partir des rendements journaliers :

$$
\Sigma = \text{Cov}(R_{\text{journaliers}}) \times 252
$$

où 252 est le nombre de jours de bourse par an.

Les rendements moyens annualisés de chaque actif sont :

$$
\mu_i = \mathbb{E}[R_{i,\text{journalier}}] \times 252
$$

### Ratio de Sharpe

Le ratio de Sharpe mesure le rendement excédentaire par unité de risque :

$$
\text{Sharpe} = \frac{\mu_{\text{portefeuille}} - r_f}{\sigma_{\text{portefeuille}}}
$$

où :
- $\mu_{\text{portefeuille}} = w^T \cdot \mu$ : rendement moyen annualisé du portefeuille
- $r_f$ : taux sans risque annualisé (calculé empiriquement à partir de l'ETF IEF - obligations américaines 7-10 ans)
- $\sigma_{\text{portefeuille}}$ : volatilité annualisée du portefeuille

Le taux sans risque empirique est calculé comme la moyenne des rendements journaliers annualisés de l'actif IEF sur la période de simulation :

$$
r_f = \mathbb{E}[R_{\text{IEF,journalier}}] \times 252
$$

---

## 4. Optimisation de portefeuille (Théorie de Markowitz)

OptInvest implémente l'optimisation quadratique de Markowitz pour trouver la répartition optimale des actifs.

### Fonction objective

On cherche à maximiser l'utilité Moyenne-Variance :

$$
U(w) = w^T \cdot \mu - \frac{\gamma}{2} \cdot w^T \cdot \Sigma \cdot w
$$

où :
- $w$ : vecteur des poids des actifs (variables à optimiser)
- $\mu$ : vecteur des rendements moyens annualisés
- $\Sigma$ : matrice de covariance annualisée des rendements
- $\gamma$ : paramètre d'aversion au risque

En pratique, on minimise $-U(w)$ avec la fonction `scipy.optimize.minimize`.

### Contraintes

1. **Contrainte budgétaire** : La somme des poids doit être égale à 1 :

$$
\sum_{i=1}^{n} w_i = 1
$$

3. **Contrainte de positivité** : Pas de vente à découvert :

$$
w_i \geq 0 \quad \forall i
$$

### Frontière efficiente

La frontière efficiente est générée en faisant varier le paramètre d'aversion au risque $\gamma$ sur un intervalle logarithmique :

$$
\gamma \in [10^{-2}, 10^{2}]
$$

Pour chaque valeur de $\gamma$, on résout le problème d'optimisation et on obtient un point $(σ_p, μ_p)$ sur la frontière.

### Portefeuille tangent (optimal)

Le portefeuille tangent est celui qui maximise le ratio de Sharpe parmi tous les portefeuilles de la frontière efficiente :

$$
w^* = \arg\max_{w \in \text{Frontière}} \frac{w^T \cdot \mu - r_f}{\sqrt{w^T \cdot \Sigma \cdot w}}
$$

C'est la solution optimale retournée par l'application.

### Hypothèses du modèle de Markowitz

- Les rendements des actifs suivent une distribution normale
- Les rendements moyens et la matrice de covariance sont constants dans le temps
- La matrice de covariance est inversible (définie positive)
- Les investisseurs sont rationnels et cherchent à maximiser leur utilité

---

## 5. Modélisation prédictive

### Modèle linéaire gaussien

OptInvest applique une régression linéaire (OLS - Ordinary Least Squares) pour prédire l'évolution future du portefeuille.

#### Modèle

On modélise la valeur du portefeuille en fonction du montant investi cumulé :

$$
V_t = \beta_0 + \beta_1 \cdot C_t + \varepsilon_t
$$

où :
- $V_t$ : valeur du portefeuille au temps $t$
- $C_t$ : montant total investi au temps $t$
- $\beta_0$ : intercept (ordonnée à l'origine)
- $\beta_1$ : pente (sensibilité de la valeur au montant investi)
- $\varepsilon_t$ : terme d'erreur (résidu)

Le modèle est ajusté avec `statsmodels.api.OLS` qui utilise la méthode des moindres carrés ordinaires.

#### Résidus

Les résidus représentent l'écart entre les valeurs observées et prédites :

$$
\varepsilon_t = V_t - (\beta_0 + \beta_1 \cdot C_t)
$$

Sous les hypothèses du modèle linéaire, les résidus doivent être :
- Centrés sur zéro : $\mathbb{E}[\varepsilon_t] \approx 0$
- Homoscédastiques (variance constante)
- Non autocorrélés
- Normalement distribués

#### Prédictions futures

Pour prédire la valeur du portefeuille sur $T$ périodes futures, on génère de nouveaux montants investis :

$$
C_{t+k} = C_t + k \cdot A
$$

où $A$ est l'apport périodique, et on prédit :

$$
\hat{V}_{t+k} = \beta_0 + \beta_1 \cdot C_{t+k}
$$

---

## 6. Métriques de qualité du modèle prédictif

### R² (Coefficient de détermination)

Mesure la proportion de variance expliquée par le modèle :

$$
R^2 = 1 - \frac{\sum_{t=1}^{n} \varepsilon_t^2}{\sum_{t=1}^{n} (V_t - \bar{V})^2}
$$

où $\bar{V}$ est la moyenne des valeurs observées.

Interprétation dans OptInvest :
- $R^2 > 0.8$ : qualité excellente
- $0.6 < R^2 \leq 0.8$ : qualité correcte  
- $R^2 < 0.6$ : qualité faible

### RMSE (Root Mean Square Error) relatif

Erreur quadratique moyenne en pourcentage :

$$
\text{RMSE}_{\text{rel}} = \frac{\sqrt{\frac{1}{n}\sum_{t=1}^{n} \varepsilon_t^2}}{\bar{V}} \times 100
$$

### Écart-type des résidus

Mesure la dispersion des erreurs de prédiction :

$$
\sigma_\varepsilon = \sqrt{\frac{1}{n-p} \sum_{t=1}^{n} \varepsilon_t^2}
$$

où $p$ est le nombre de paramètres du modèle (ici $p=2$).

### Statistique de Durbin-Watson

Teste l'autocorrélation des résidus :

$$
DW = \frac{\sum_{t=2}^{n} (\varepsilon_t - \varepsilon_{t-1})^2}{\sum_{t=1}^{n} \varepsilon_t^2}
$$

Interprétation :
- $DW \approx 2$ : pas d'autocorrélation
- $DW < 2$ : autocorrélation positive
- $DW > 2$ : autocorrélation négative

### Analyse des coefficients

- **P-values** : Probabilité que le coefficient soit nul (significativité)
  - Si $p < 0.05$, le coefficient est statistiquement significatif

- **Intervalles de confiance à 95%** : Plage de valeurs plausibles pour chaque coefficient

### Analyse d'overfitting

Pour détecter le surapprentissage, on divise les données en :
- **Ensemble d'entraînement** : 80% des données
- **Ensemble de validation** : 20% des données

On calcule la différence entre les R² :

$$
\Delta R^2 = |R^2_{\text{train}} - R^2_{\text{val}}|
$$

Un $\Delta R^2$ élevé indique un surapprentissage.

---

## 7. Modèles prédictifs futurs (en développement)

Le code inclut des classes pour des modèles plus avancés (non encore implémentés) :

### Processus autorégressif (AR)

Modélise la valeur comme une combinaison linéaire de ses valeurs passées :

$$
V_t = c + \sum_{i=1}^{p} \phi_i V_{t-i} + \varepsilon_t
$$

### Modèle ARMA

Combine autorégressif et moyenne mobile :

$$
V_t = c + \sum_{i=1}^{p} \phi_i V_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t
$$

### Modèle ARMA-GARCH

Modélise à la fois la moyenne (ARMA) et la volatilité conditionnelle (GARCH) :

$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$

---

## 8. Synthèse

OptInvest combine :

1. **Simulation historique** : Calcul de la performance réelle des stratégies DCA et Lump Sum
2. **Optimisation de Markowitz** : Recherche de la répartition optimale des actifs maximisant le ratio de Sharpe
3. **Modélisation prédictive** : Régression linéaire avec analyse statistique complète des résidus
4. **Métriques financières** : CAGR, volatilité, ratio de Sharpe basés sur la théorie moderne de portefeuille

Toutes les formules sont implémentées dans les modules `simulator.py` et `predictor.py` en utilisant NumPy, pandas, statsmodels et scipy.
