from datetime import date, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.schemas.simulation import ACTIFS, Frequency

class PredictionRequest(BaseModel):
    actifs: List[str] = Field(
        ...,
        description="Sélectionnez 1 à 6 actifs autorisés",
        examples=[["AAPL"]],
        min_length=1,
    )
    duree_ans: int = Field(
        ...,
        ge=1,
        description="Durée de la simulation historique en années",
        examples=[6],
    )
    duree_prediction_ans: int = Field(
        ...,
        ge=1,
        le=10,
        description="Durée de prédiction en années",
        examples=[5],
    )
    date_debut: Optional[date] = Field(
        None,
        description="Date de départ (remplie automatiquement si omise)",
    )
    montant_initial: float = Field(
        ...,
        gt=0,
        description="Capital initial (€)",
        examples=[10_000.0],
    )
    apport_periodique: float = Field(
        ...,
        ge=0,
        description="Montant de l'apport périodique (€)",
        examples=[200.0],
    )
    frais_gestion: float = Field(
        0.2,
        ge=0,
        description="Frais de gestion annuels (%)",
        examples=[0.2],
    )

    # --------- VALIDATION ---------
    @field_validator("actifs", mode="before")
    @classmethod
    def _verifier_actifs(cls, v):
        v = [v] if isinstance(v, str) else v
        v = [t.upper().strip() for t in v]
        inconnus = set(v) - ACTIFS
        if inconnus:
            raise ValueError(
                f"Actifs non autorisés : {', '.join(inconnus)}. "
                f"Choix possibles : {', '.join(sorted(ACTIFS))}."
            )
        return v
    
    @model_validator(mode="after")
    def _set_default_date(self):
        if self.date_debut is None:
            self.date_debut = date.today() - timedelta(days=365 * self.duree_ans)
        return self

class RegressionMetrics(BaseModel):
    r2: float = Field(..., description="Coefficient de détermination")
    rmse: float = Field(..., description="Erreur quadratique moyenne")
    mae: float = Field(..., description="Erreur absolue moyenne")
    std_residus: float = Field(..., description="Écart-type des résidus")
    pente: float = Field(..., description="Pente de la droite de régression")
    intercept: float = Field(..., description="Ordonnée à l'origine")

class StrategyResult(BaseModel):
    timeline_historique: Dict[str, float] = Field(..., description="Timeline historique")
    predictions: Dict[str, float] = Field(..., description="Prédictions futures")
    regression_metrics: RegressionMetrics = Field(..., description="Métriques de qualité")
    cagr: float = Field(..., description="CAGR de la stratégie")
    rendement_total: float = Field(..., description="Rendement total")

class PredictionResponse(BaseModel):
    strategies: Dict[str, StrategyResult] = Field(
        ..., 
        description="Résultats pour chaque stratégie"
    )