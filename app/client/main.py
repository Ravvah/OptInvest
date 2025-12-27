from __future__ import annotations
from typing import List, Tuple

import streamlit as st
from loguru import logger

from app.api.schemas.prediction import PredictionResponse
from app.api.schemas.simulation import SimulationRequest, SimulationResponse
from app.client.user_form_manager import UserForm
from app.client.api_manager import APIManager
from app.client.plot_manager import DashboardManager
import app.client.constants as cst

class ClientApp:
    
    def __init__(self, user_interface: UserForm, api_manager: APIManager, graphique_manager: DashboardManager) -> None:
        self.user_interface = user_interface
        self.api_manager = api_manager
        self.graphique_manager = graphique_manager
        # self.export_manager = export_manager

    def run(self) -> None:
        """
        Lance l'interface principale de l'application.
        """
        st.set_page_config(page_title="OPTINVEST", page_icon="📈", layout="wide")
        st.title("OPTINVEST - Simulation de portefeuille")

        parametres_simulation_dict = self.user_interface.display_user_form_sidebar()
        api_simulation_requete, duree_prediction_ans = self.user_interface._build_portfolio_simulation_request(parametres_simulation_dict) if parametres_simulation_dict else (None, None)
         
        if api_simulation_requete and duree_prediction_ans:
            self._execute_portfolio_simulation_with_user_parameters(api_simulation_requete, duree_prediction_ans)

    def _execute_portfolio_simulation_with_user_parameters(self, api_simulation_request: SimulationRequest, duree_prediction_ans: str) -> None:
        """
        Exécute la simulation avec les paramètres donnés.
        """
        with st.spinner("Simulation en cours..."):
            donnees_simulation_portefeuille = self.api_manager.call_api_simulation(parametres_requete=api_simulation_request)
            parametres_indice_ref = SimulationRequest(
                actifs=[cst.INDICE_REF],
                **api_simulation_request.model_dump(mode="json", exclude={"actifs"}))
            valeur_portefeuille_temps_indice_ref = self.api_manager.call_api_simulation(parametres_indice_ref).valeur_portefeuille_temps
            logger.info(" -------------- Paramètres de la simulation ACIM ---------------")
            logger.info(parametres_indice_ref.model_dump_json())

            api_prediction_requete = self.user_interface._build_prediction_request(valeur_portefeuille_montant=donnees_simulation_portefeuille.valeur_portefeuille_montant,
                                                                                   valeur_portefeuille_temps=donnees_simulation_portefeuille.valeur_portefeuille_temps,
                                                                                   duree_prediction_ans=duree_prediction_ans,
                                                                                   apport_periodique=api_simulation_request.apport_periodique or 0.0,
                                                                                   frequence=api_simulation_request.frequence,
                                                                                   modele=cst.MODELE_LINEAIRE)                         
            donnees_prediction_portefeuille = self.api_manager.call_api_prediction(parametres_requete=api_prediction_requete)

            donnees_simulation_prediction_portefeuille = self.api_manager.simulate_portfolio_for_all_strategies(parametres_simulation=api_simulation_request, duree_prediction_ans=duree_prediction_ans)
            
            self.display_results(donnees_simulation_portefeuille=donnees_simulation_portefeuille, donnees_prediction_portefeuille=donnees_prediction_portefeuille, valeur_portefeuille_temps_indice_ref=valeur_portefeuille_temps_indice_ref, donnees_simulation_prediction_portefeuille=donnees_simulation_prediction_portefeuille)
       
    def display_results(self, donnees_simulation_portefeuille: SimulationResponse,  donnees_prediction_portefeuille: PredictionResponse, valeur_portefeuille_temps_indice_ref: dict[str, float], donnees_simulation_prediction_portefeuille = List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        """
        Affiche tous les résultats et options d'export.
        """
        st.session_state.donnees_simulation_portefeuille = donnees_simulation_portefeuille
        st.session_state.valeur_portefeuille_temps_indice_ref = valeur_portefeuille_temps_indice_ref
        st.session_state.donnees_prediction_portefeuille = donnees_prediction_portefeuille
        st.session_state.donnees_simulation_prediction_portefeuille = donnees_simulation_prediction_portefeuille
        
        st.success("Simulation terminée !")
        
        self.graphique_manager.display_principal_metrics(donnees_simulation_portefeuille)
        self.graphique_manager.display_portfolio_vs_cash_plot(donnees_simulation_portefeuille)
        
        if valeur_portefeuille_temps_indice_ref is not None and len(valeur_portefeuille_temps_indice_ref) > 0:
            self.graphique_manager.display_portfolio_vs_world(donnees_simulation_portefeuille, valeur_portefeuille_temps_indice_ref)
        
        self.graphique_manager.display_mensual_returns_distribution(donnees_simulation_portefeuille)
        self.graphique_manager.display_all_models_predictions(donnees_simulation_prediction_portefeuille=donnees_simulation_prediction_portefeuille)
        self.graphique_manager.display_prediction_data(donnees_simulation=donnees_simulation_portefeuille, donnees_prediction=donnees_prediction_portefeuille)
        self.graphique_manager.display_markowitz_optimal_repartition(donnees_simulation=donnees_simulation_portefeuille)
        
        # self.export.afficher_options_export(
        #     donnees_simulation_portefeuille, 
        #     donnees_prediction_portefeuille, 
        #     valeur_portefeuille_temps_indice_ref
        # )

if __name__ == "__main__":
    user_interface = UserForm()
    api_manager = APIManager()
    graphique_manager = DashboardManager()
    ClientApp(user_interface=user_interface, api_manager=api_manager, graphique_manager=graphique_manager).run()