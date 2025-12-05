from datetime import date, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import app.api.schemas.constants as cst

class SimulationRequest(BaseModel):
    actifs: List[str] = Field(
        ...,
        description="Sélectionnez 1 à 6 actifs autorisés",
        examples=[["AAPL"]],
        min_length=1,
        max_length=6
    )
    # La date de départ est fixée à 'aujourd'hui - duree_ans' par défaut


    duree_ans: int = Field(
        ...,
        ge=1,
        description="Durée de la simulation en années",
        examples=[6],
    )

    date_debut: Optional[date] = Field(
        None,
        description="Date de départ (remplie automatiquement si omise)",
        examples=["2018-01-01"]
    )

    date_fin: Optional[date] = Field (
        None,
        description="Date de fin (remplie automatiquement si omise)",
    )
    
    montant_initial: float = Field(
        ...,
        gt=0,
        description="Capital initial (€)",
        examples=[10_000.0],
    )
    apport_periodique: Optional[float] = Field(
        ...,
        ge=0,
        description="Montant de l’apport périodique (€)",
        examples=[200.0],
    )
    strategie: cst.STRATEGIE = Field(
        default="dca", description="Stratégie d'investissement"
    )
    frequence: Optional[cst.FREQUENCY] = Field(
        ...,
        description="Fréquence des apports",
        examples=["mensuelle"],
    )
    frais_gestion: float = Field(
        0.2,
        ge=0,
        description="Frais de gestion annuels (%)",
        examples=[0.2],
    )

    # ---------- VALIDATIONS ----------
    @field_validator("actifs", mode="before")
    @classmethod
    def _verifier_actifs(cls, v):
        """Convertit 'actif' → ['actif'] et vérifie la whitelist."""
        v = [v] if isinstance(v, str) else v
        v = [t.upper().strip() for t in v]
        inconnus = set(v) - cst.ACTIFS_DISPONIBLES
        if inconnus:
            raise ValueError(
                f"Actifs non autorisés : {', '.join(inconnus)}. "
                f"Choix possibles : {', '.join(sorted(cst.ACTIFS_DISPONIBLES))}."
            )
        return v
    
    @model_validator(mode="after")
    def _set_default_date(self):
        """Si date_debut est pas spécifiée → today - durée."""
        if self.date_debut is None:
            self.date_debut = date.today() - timedelta(days=365 * self.duree_ans)
        return self
    
    @model_validator(mode="after")
    def _verify_strategie_coherency(self):
        """Verifier que si lump sum est la strategie, frequence et apport est vide"""
        if self.strategie == "lump sum":
            self.frequence = None 
            self.apport_periodique = None
        return self
    

class MarkowitzOptimalResult(BaseModel):
    repartition_actifs: Dict[str, float] = Field(..., description="Poids optimaux au sens de Markowitz des actifs choisis", examples=[{
        "ACIM": 0.6,
        "AAPL": 0.4
    }])
    ratio_sharpe: float = Field(..., description="Ratio de Sharpe maximal de la frontière efficiente")
    rendement_moyen: float = Field(..., description="Rendement moyen pour la répartition optimale des actifs")
    volatilite_annualisee: float = Field(..., description="Volatilite des rendements pour la répartition optimale des actifs")


class SimulationResponse(BaseModel):
    frequence: Optional[cst.FREQUENCY] = Field(
        ...,
        description="Fréquence des apports",
        examples=["mensuelle"],
    )
    cash_investi: float = Field(..., description="Total du cahs investi dans la période d'investissement")
    cagr: float = Field(..., description="Taux de croissance annuel composé")
    rendement_total: float = Field(..., description="Rendement total (€) pour une répartition égale des actifs")
    volatilite_annualisee: float = Field(..., description="Volatilité annualisée")
    ratio_sharpe: float = Field(..., description="Ratio de Sharpe pour une répartition égale des actifs")
    repartition_optimale_markowitz: MarkowitzOptimalResult = Field(..., description="Répartition optimale des actifs choisis selon le modèle Markowitz")
    valeur_portefeuille_temps: Dict[str, float] = Field(
        ...,
        description="Valeur du portefeuille par date ISO",
        examples=[{"2024-05-31": 10450.0}],
    )
    valeur_portefeuille_montant: Dict[str, float] = Field(
        ..., 
        description="Valeur du portefeuille par montant investi",
        examples=[{"1500": 3500.5}]
    )
    valeur_montant_temps: Dict[str, float] = Field(..., description="Montants investis par date ISO",
    examples=[{"2024-05-31": 1500}]
    )
    valeur_rendements_mensuels_temps: Dict[str, float] = Field(..., description="Valeur rendements mensuels du portefeuille par date ISO")
    

