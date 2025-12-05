from typing import Dict, List

API_URL = "http://localhost:8000/api/simuler"
API_PREDICTION_URL = "http://localhost:8000/api/predire"
INDICE_REF = "ACIM"
ACTIFS_AUTORISES: List[str] = [
    "AAPL", "MSFT", "AGGH", "TLT", "VWCE.DE", "SXR8.DE"
]

FREQUENCES: Dict[str, str] = {
    "Mensuelle": "mensuelle",
    "Trimestrielle": "trimestrielle", 
    "Semestrielle": "semestrielle",
    "Annuelle": "annuelle"
}

MOIS_PAR_FREQ: Dict[str, int] = {
    "mensuelle": 1,
    "trimestrielle": 3,
    "semestrielle": 6,
    "annuelle": 12,
}
MODELE_LINEAIRE = "modele_lineaire"

COULEURS_PALETTE = ['#FF6B35', '#004E89', '#009639', '#7209B7', '#D62828', '#F77F00']