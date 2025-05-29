from datetime import date, timedelta
from typing import List, Literal, Dict

from pydantic import BaseModel, Field, field_validator, model_validator

Frequency = Literal["mensuelle", "trimestrielle", "semestrielle", "annuelle"]

ACTIFS: set[str] = {
    "AAPL",  # Action
    "MSFT",  # Action
    "AGGH",  # ETF obligataire global
    "TLT",   # ETF obligations US long terme
    "VWCE",  # ETF actions monde
    "SXR8",  # ETF S&P 500
}


class SimulationRequest(BaseModel):
    actifs: List[str] = Field(
        ...,
        description="Sélectionnez 1 à 6 actifs autorisés",
        examples=[["VWCE", "AGGH"]],
        min_length=1,
    )
    # La date de départ est fixée à 'aujourd’hui' par défaut
    date_debut: date = Field(
        default_factory=date.today,
        description="Date de départ (fixée à aujourd’hui)",
        examples=[date.today().isoformat()],
    )
    duree_ans: int = Field(
        ...,
        ge=1,
        description="Durée de la simulation en années",
        examples=[6],
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
        description="Montant de l’apport périodique (€)",
        examples=[200.0],
    )
    frequence: Frequency = Field(
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
        inconnus = set(v) - ACTIFS
        if inconnus:
            raise ValueError(
                f"Actifs non autorisés : {', '.join(inconnus)}. "
                f"Choix possibles : {', '.join(sorted(ACTIFS))}."
            )
        return v


class SimulationResponse(BaseModel):
    cagr: float = Field(..., description="Taux de croissance annuel composé")
    rendement_total: float = Field(..., description="Rendement total (€)")
    timeline: Dict[str, float] = Field(
        ...,
        description="Valeur du portefeuille par date ISO",
        examples=[{"2024-05-31": 10450.0}],
    )
