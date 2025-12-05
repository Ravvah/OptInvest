from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

import app.api.schemas.constants as cst


class PredictionRequest(BaseModel):
    x: List[List[float]] = Field(
        ...,
        description="Observations des variables indépendantes (montants investis sur la période choisie, volume journalier de l'actif)",
    )

    y: List[float] = Field(
        ..., description= "Observations de la variable dépendante (valeur du portefeuille)"
    )

    duree_prediction_ans: int = Field(
        ...,
        ge=1,
        le=10,
        description="Durée de prédiction en années",
        examples=[5],
    )

    apport_periodique: float = Field(
        ..., description="Montant investi à chaque période"
    )

    frequence: Optional[cst.FREQUENCY] = Field(
        ..., description="Fréquence des apports"
    )
    
    date_derniere_simulation: date = Field(
        ..., description="Derniere date de l'historique de la valeur du portefeuille"
    )

    modele: str = Field(
        ...,
        description="Modèle de prédiction à utiliser",
        examples=["modele_lineaire", "processus_gaussien"],)
    
    @model_validator(mode="after")
    def _verify_shape(self):
        if len(self.x) != len(self.y):
            raise ValueError("La variable dépendante y et les variables indépendantes X doivent avoir la même taille !")
        if len(self.x) < 5:
            raise ValueError("Nombre de données x insuffisantes, il faut au moins 5 observations !")
        return self

    
    class Config:
        json_schema_extra = {
            "example": {
                "x": [[10000.0], [10200.0], [10400.0], [10600.0], [10800.0], [11000.0], [11200.0]],
                "y": [130000.6, 170000.3, 190000.9, 250000.3, 270000.4, 299000.5, 310000.7],
                "duree_prediction_ans": 3,
                "date_derniere_simulation": str(datetime.now().date()),
                "apport_periodique": 200,
                "frequence": "mensuelle",
                "modele": "modele_lineaire"
            }
        }


# class LinearRegressionMetrics(BaseModel):
#     r2: float = Field(..., description="Coefficient de détermination")
#     rmse: float = Field(..., description="Erreur quadratique moyenne")
#     mae: float = Field(..., description="Erreur absolue moyenne")
#     std_residus: float = Field(..., description="Écart-type des résidus")
#     intercept: float = Field(..., description="Ordonnée à l'origine")
#     dw_stat: float = Field(..., description="Statistique Durbin-Watson pour l'autocorrélation des résidus")


# class StochasticProcessMetrics(BaseModel):
#     aic: float = Field(..., description="AIC du modèle AR")
#     bic: float = Field(..., description="BIC du modèle AR")
#     params: Dict[str, float] = Field(..., description="Paramètres du modèle AR")

# class PredictionResult(BaseModel):
#     timeline_historique: Dict[str, float] = Field(..., description="Timeline historique")
#     predictions: Dict[str, float] = Field(..., description="Prédictions futures")
#     regression_metrics: Optional[LinearRegressionMetrics] = Field(None, description="Métriques de qualité (pour régression)")
#     gp_metrics: Optional[StochasticProcessMetrics] = Field(None, description="Métriques du modèle AR")
#     residus: Optional[List[float]] = Field(None, description="Résidus de la régression")
#     volume: Optional[List[float]] = Field(None, description="Données de volume (pour modèle temps+volume)")
#     cagr: float = Field(..., description="CAGR de la stratégie")
#     rendement_total: float = Field(..., description="Rendement total")


class FitQualityAnalysis(BaseModel):
    r2: float = Field(..., description="Coefficient de détermination R² du modèle")
    rmse: float= Field(..., description="Erreur quadratique moyenne du modèle")

class CoefficientAnalysis(BaseModel):
    coeff_p_values: Dict[str, float] = Field(
        ..., description="P-valeurs des coefficients du modèle pour la significativité"  
    )
    coeff_confidence_intervals: Dict[str, List[float]] = Field(
        ..., description="Intervalles de confiances des coefficients estimés du modèle"
    )

class OverfittingAnalysis(BaseModel):
    r2_diff: float = Field(..., description="Différence R² entre entraînement et validation")

class ResidualAnalysis(BaseModel):
    residuals: List[float] = Field(..., description="Liste des résidus du modèle")
    moyenne: float = Field(..., description="Moyenne des résidus")
    ecart_type: float = Field(..., description="Variance des résidus")
    autocorrelation_statistic: float = Field(..., description="Statistique d'autocorrélation des résidus de Durbin-Watson")

class PredictionResponse(BaseModel):
    model: cst.MODELE = Field(..., description="Modèle utilisé pour la prédiction")
    predictions: Dict[str, float] = Field(..., description="Liste des points de prédiction par montants investis futur")
    predictions_timeline: Dict[str, float] = Field(..., description="Liste des points de prédictions par date future (chaque fin du mois)")
    fit_quality_analysis: FitQualityAnalysis = Field(..., description="Analyse de la qualité d'ajustement du modèle")
    coeff_analysis: CoefficientAnalysis = Field(..., description="Analyse des coefficients du modèle")
    residual_analysis: ResidualAnalysis = Field(..., description="Analyse des résidus du modèle")
    overfitting_analysis: Optional[OverfittingAnalysis] = Field(None, description="Analyse du surapprentissage")



