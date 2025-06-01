from __future__ import annotations
from typing import Dict

import pandas as pd
import requests

class GestionnaireApi:
    API_URL = "http://localhost:8000/api/simuler"
    API_PREDICTION_URL = "http://localhost:8000/api/predire"
    INDICE_REF = "ACIM"

    def appeler_simulation(self, parametres_requete: Dict) -> Dict:
        """
        Appelle l'API de simulation.
        """
        reponse_api = requests.post(self.API_URL, json=parametres_requete, timeout=30)
        reponse_api.raise_for_status()
        return reponse_api.json()

    def appeler_prediction(self, parametres_requete: Dict) -> Dict:
        """
        Appelle l'API de prédiction.
        """
        reponse_api = requests.post(self.API_PREDICTION_URL, json=parametres_requete, timeout=30)
        reponse_api.raise_for_status()
        return reponse_api.json()

    def charger_benchmark(self, parametres_base: Dict, montant_initial: float) -> pd.Series:
        """
        Charge les données de benchmark via l'API.
        """
        parametres_benchmark = {
            **parametres_base,
            "actifs": [self.INDICE_REF],
        }
        
        try:
            donnees_benchmark = self.appeler_simulation(parametres_benchmark)
        except requests.exceptions.RequestException:
            return pd.Series(dtype=float)

        serie_temporelle = pd.Series(donnees_benchmark["timeline"], dtype=float)
        serie_temporelle.index = pd.to_datetime(serie_temporelle.index)
        serie_temporelle = serie_temporelle / serie_temporelle.iloc[0] * montant_initial
        serie_temporelle.name = "Indice ACWI IMI"
        
        return serie_temporelle