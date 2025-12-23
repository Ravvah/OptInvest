from __future__ import annotations

from typing import List, Tuple
import requests
from loguru import logger

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.api.schemas.simulation import SimulationRequest, SimulationResponse
from app.client.user_form_manager import UserForm
import app.client.constants as cst

class APIManager:

    def __init__(self):
        self.ui_manager = UserForm()

    def call_api_simulation(self, parametres_requete: SimulationRequest) -> SimulationResponse:
        """
        Appelle l'API de simulation.
        """
        logger.info("PARAMETRES SIMULATION")
        logger.info(parametres_requete)
        reponse_api = requests.post(url=cst.API_URL, json=parametres_requete.model_dump(mode="json"), timeout=30)
        if reponse_api.status_code != 200 or reponse_api.text is None:
            raise Exception(f"Erreur API: {reponse_api.status_code} - {reponse_api.text}")

        return SimulationResponse(**reponse_api.json())


    def call_api_prediction(self, parametres_requete: PredictionRequest) -> PredictionResponse:
        """
        Appelle l'API de prédiction.
        """

        reponse_api = requests.post(cst.API_PREDICTION_URL, json=parametres_requete.model_dump(mode="json"), timeout=30)
        if reponse_api.status_code != 200 or reponse_api.text is None:
            raise Exception(f"Erreur API: {reponse_api.status_code} - {reponse_api.text}")

        return PredictionResponse(**reponse_api.json())


    def simulate_portfolio_for_all_strategies(self, parametres_simulation: SimulationRequest, duree_prediction_ans:int) -> List[Tuple[SimulationResponse, PredictionResponse]]:
        result_all_strategies = []
        for frequence in list(cst.FREQUENCES.values()):
            parametres_simulation.frequence = frequence
            resultat_simulation = self.call_api_simulation(parametres_requete=parametres_simulation)
            parametres_prediction = self.ui_manager._build_prediction_request(valeur_portefeuille_temps=resultat_simulation.valeur_portefeuille_temps, valeur_portefeuille_montant=resultat_simulation.valeur_portefeuille_montant, duree_prediction_ans=duree_prediction_ans, apport_periodique=parametres_simulation.apport_periodique, frequence=frequence, modele=cst.MODELE_LINEAIRE)
            resultat_prediction = self.call_api_prediction(parametres_requete=parametres_prediction)
            result_all_strategies.append((resultat_simulation, resultat_prediction))
        parametres_simulation_lump_sum = SimulationRequest(
                frequence="lump sum",
                **parametres_simulation.model_dump(mode="json", exclude={"frequence"}))
        resultat_simulation_lump_sum = self.call_api_simulation(parametres_requete=parametres_simulation_lump_sum)
        parametres_prediction_lump_sum = self.ui_manager._build_prediction_request(valeur_portefeuille_temps=resultat_simulation_lump_sum.valeur_portefeuille_temps, valeur_portefeuille_montant=resultat_simulation_lump_sum.valeur_portefeuille_montant, duree_prediction_ans=duree_prediction_ans, apport_periodique=parametres_simulation_lump_sum.apport_periodique, frequence=None, modele=cst.MODELE_LINEAIRE)
        resultat_prediction_lump_sum = self.call_api_prediction(parametres_requete=parametres_prediction_lump_sum)
        result_all_strategies.append((resultat_simulation_lump_sum, resultat_prediction_lump_sum))
        return result_all_strategies
