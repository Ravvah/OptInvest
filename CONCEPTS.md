# Concepts utilisés dans OptInvest

## 1. Objectif

OptInvest compare deux stratégies d’investissement :  
- **DCA (Dollar-Cost Averaging)** : investir progressivement à intervalles réguliers.  
- **Lump Sum** : investir tout le capital en une seule fois.  
Le projet simule l’évolution du portefeuille selon chaque approche, puis analyse et prédit la performance.

---

## 2. Stratégies d’investissement

### DCA (investissement progressif)

À chaque période $k$, on investit un montant $c_k$ lorsque le prix de l’actif est $P_{t_k}$.  
Le nombre de parts achetées est :

$$
s_k = \frac{c_k}{P_{t_k}}
$$

- $s_k$ : nombre de parts achetées à la période $k$
- $c_k$ : montant investi à la période $k$
- $P_{t_k}$ : prix de l’actif à la période $t_k$

La valeur finale du portefeuille au temps $T$ est :

$$
V_T^{\text{DCA}} = \sum_{k=0}^{n-1} s_k \cdot P_T = \sum_{k=0}^{n-1} c_k \frac{P_T}{P_{t_k}}
$$

- $V_T^{\text{DCA}}$ : valeur du portefeuille à la date finale $T$
- $P_T$ : prix de l’actif à la date finale $T$

---

### Lump Sum (investissement unique)

On investit un montant total $C$ au début, lorsque le prix est $P_{t_0}$.  
La valeur finale est :

$$
V_T^{\text{LS}} = C \frac{P_T}{P_{t_0}}
$$

- $V_T^{\text{LS}}$ : valeur du portefeuille à la date finale $T$
- $C$ : montant investi au départ
- $P_{t_0}$ : prix de l’actif au moment de l’investissement

---

## 3. Mesures de performance

Pour comparer les stratégies, OptInvest calcule :

#### - **Rendement total** : 
Différence entre la valeur finale et le montant investi.
#### - **CAGR (taux de croissance annuel composé)** :

$$
\text{CAGR} = \left( \frac{V_f}{V_i} \right)^{1/T} - 1
$$

- $V_f$ : valeur finale du portefeuille
- $V_i$ : montant total investi
- $T$ : durée en années

#### - **Volatilité annualisée** : 
Mesure la dispersion des rendements périodiques, calculée comme l’écart-type des rendements multiplié par la racine du nombre de périodes par an.

Mathématiquement, si $r_1, r_2, \ldots, r_n$ sont les rendements périodiques (par exemple mensuels), l’écart-type des rendements est :

$$
\sigma_r = \sqrt{ \frac{1}{n} \sum_{i=1}^n (r_i - \bar{r})^2 }
$$

où $\bar{r}$ est le rendement moyen :

$$
\bar{r} = \frac{1}{n} \sum_{i=1}^n r_i
$$

La volatilité annualisée est alors :

$$
\sigma_{\text{ann}} = \sigma_r \times \sqrt{f}
$$

où $f$ est le nombre de périodes par an (par exemple $f=12$ pour des rendements mensuels).

- $\sigma_r$ : écart-type des rendements périodiques
- $\sigma_{\text{ann}}$ : volatilité annualisée
- $f$ : nombre de périodes par an

#### - **Ratio de Sharpe** :

Un ratio de Sharpe positif et élevé indique que la stratégie offre un rendement supérieur au taux sans risque, pour un niveau de volatilité donné.

$$
\text{Sharpe} = \frac{\mu - r_f}{\sigma}
$$

- $\mu$ : rendement moyen annualisé du portefeuille
- $r_f$ : taux sans risque annualisé
- $\sigma$ : volatilité annualisée

---

## 4. Modélisation et prédiction

Pour estimer la tendance future, on va étudier plusieurs approches de modélisation temporelle. La plus simple et la première sera le modèle linéaire gaussien. On aborde aussi les modélisations par series temporelles.

### 1) Modèle linéaire gaussien

OptInvest applique une régression linéaire sur la série historique de la valeur du portefeuille. Cette approche permet d’obtenir une projection simple de l’évolution du portefeuille dans le temps, en prolongeant la tendance observée sur la période simulée.

La régression est réalisée sur les valeurs historiques, où chaque date est transformée en nombre de jours écoulés depuis le début de la simulation. Le modèle ajuste une droite aux données :

$$
V_t = a + b t + \varepsilon_t
$$

- $V_t$ : valeur du portefeuille à la date $t$
- $a$ : intercept (valeur initiale estimée)
- $b$ : pente (croissance estimée par unité de temps)
- $\varepsilon_t$ : erreur de prédiction à la date $t$

Les résidus $\varepsilon_t$ sont définis par :

$$
\varepsilon_t = V_t - (a + b t)
$$

Ils représentent l'écart entre la valeur observée et la valeur prédite par le modèle.

OptInvest analyse la distribution des résidus pour vérifier la validité du modèle :
- **Écart-type des résidus** :

$$
\sigma_\varepsilon = \sqrt{\frac{1}{n} \sum_{t=1}^n \varepsilon_t^2}
$$

où $n$ est le nombre de points simulés.

- **Histogramme et densité** : la forme des résidus est comparée à une loi normale centrée sur zéro.
- **Absence de biais** : on vérifie que la moyenne des résidus est proche de zéro :

$$
\bar{\varepsilon} = \frac{1}{n} \sum_{t=1}^n \varepsilon_t \approx 0
$$

Une forte dispersion ou une structure particulière dans les résidus peut indiquer que le modèle linéaire n’explique pas correctement la dynamique du portefeuille.

Pour évaluer la qualité de la prédiction, plusieurs métriques sont calculées :
- **$R^2$** : part de la variance expliquée par le modèle (plus il est proche de 1, meilleure est la prédiction)
- **RMSE** : racine de l’erreur quadratique moyenne, mesure la précision globale
- **MAE** : erreur absolue moyenne
- **Écart-type des résidus** : mesure la dispersion des erreurs
- **Pente et intercept** : permettent d’interpréter la tendance estimée

OptInvest propose aussi une visualisation de la distribution des résidus, comparée à une loi normale, pour aider à juger la validité du modèle. Ces diagnostics sont affichés dans l’interface et exportés dans les rapports.

![image](https://github.com/user-attachments/assets/0e21d11e-4d82-41b7-8d21-7575fde7e40d)

Enfin, la qualité globale du modèle est synthétisée dans l’UI :  
- $R^2 > 0.8$ : excellente qualité  
- $0.6 < R^2 \leq 0.8$ : qualité correcte  
- $R^2 < 0.6$ : qualité faible

---

## 5. Synthèse

OptInvest exploite ces concepts pour simuler, comparer et prédire la performance des stratégies DCA et Lump Sum, en s’appuyant sur des formules simples et des indicateurs standards du monde financier.
