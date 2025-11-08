from __future__ import annotations
from typing import Dict

import pandas as pd
import requests
import numpy as np

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
        Charge l'indice de référence via simulation DCA (même stratégie que le portefeuille).
        """
        parametres_benchmark = {
            **parametres_base,  # Mêmes dates, montant initial, apports, fréquence
            "actifs": [self.INDICE_REF]  # Mais uniquement ACIM
        }
        
        try:
            donnees_benchmark = self.appeler_simulation(parametres_benchmark)
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du chargement de l'indice {self.INDICE_REF}: {e}")
            return pd.Series(dtype=float)

        # Créer la série
        serie = pd.Series(donnees_benchmark["timeline"], dtype=float)
        serie.index = pd.to_datetime(serie.index)
        serie = serie.sort_index()
        
        # NE PAS supprimer les valeurs - l'API retourne déjà les bonnes valeurs
        # La série doit commencer au montant_initial comme le portefeuille
        
        if serie.empty:
            print(f"Aucune donnée valide pour l'indice {self.INDICE_REF}")
            return pd.Series(dtype=float)
        
        serie.name = f"Indice {self.INDICE_REF}"
        
        print(f"Benchmark {self.INDICE_REF} chargé : {len(serie)} points")
        print(f"Début: {serie.iloc[0]:.2f}€, Fin: {serie.iloc[-1]:.2f}€")
        
        return serie