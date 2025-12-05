from typing import Literal

ACTIFS_DISPONIBLES: set[str] = {
    "AAPL",  # Action
    "MSFT",  # Action
    "AGGH",  # ETF obligataire global
    "TLT",   # ETF obligations US long terme
    "VWCE.DE",  # ETF actions monde
    "SXR8.DE",  # ETF S&P 500
    "ACIM",  # ETF actions monde
}

ACTIFS_A_CONVERTIR = Literal["AAPL", "MSFT", "TLT"]
FREQUENCY = Literal["mensuelle", "trimestrielle", "semestrielle", "annuelle", "lump sum"]
STRATEGIE = Literal["dca", "lump sum"]
FREQ2MONTH = {"mensuelle": 1, "trimestrielle": 3, "semestrielle": 6, "annuelle": 12, "lump sum": 0}
ACTIF_SANS_RISQUE = "IEF"
ACTIF_CONVERSION_DOLLARD_EURO = "EURUSD=X"
NOMBRE_JOURS_BOURSE_AN = 252

MODELE = Literal["modele_lineaire", "processus_autoregressif"]