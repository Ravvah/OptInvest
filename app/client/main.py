from __future__ import annotations

import streamlit as st
import requests

from ui_manager import InterfaceUtilisateur
from api_manager import GestionnaireApi
from plots_manager import GestionnaireGraphiques
from export_manager import GestionnaireExport

class OptinvestApp:
    
    def __init__(self) -> None:
        self.interface = InterfaceUtilisateur()
        self.api = GestionnaireApi()
        self.graphiques = GestionnaireGraphiques()
        self.export = GestionnaireExport()

    def run(self) -> None:
        """
        Lance l'interface principale de l'application.
        """
        st.set_page_config(page_title="OPTINVEST", page_icon="📈", layout="wide")
        st.title("OPTINVEST – Simulation de portefeuille")

        parametres_simulation = self.interface.afficher_sidebar()
        
        if parametres_simulation:
            self._executer_simulation_avec_parametres(parametres_simulation)

    def _executer_simulation_avec_parametres(self, parametres: dict) -> None:
        """
        Exécute la simulation avec les paramètres donnés.
        """
        try:
            with st.spinner("Simulation en cours..."):
                donnees_portefeuille = self.api.appeler_simulation(parametres["base"])
                serie_benchmark = self.api.charger_benchmark(parametres["base"], montant_initial=parametres["montant_initial"])
                print(" -------------- Paramètres de la simulation ACIM ---------------")
                print(parametres)
                print("---------- Benchmark loaded in client main.py -------")
                print(serie_benchmark)
                donnees_prediction = self.api.appeler_prediction(parametres["prediction"])
                
                self._afficher_resultats(donnees_portefeuille, serie_benchmark, donnees_prediction, parametres)
                
        except Exception as e:
            st.error(f"Erreur lors de la simulation: {str(e)}")

    def _afficher_resultats(self, donnees_portefeuille: dict, serie_benchmark, donnees_prediction: dict, parametres: dict) -> None:
        """
        Affiche tous les résultats et options d'export.
        """
        st.session_state.donnees_portefeuille = donnees_portefeuille
        st.session_state.serie_benchmark = serie_benchmark
        st.session_state.donnees_prediction = donnees_prediction
        st.session_state.parametres = parametres
        
        st.success("Simulation terminée !")
        
        self.graphiques.afficher_metriques_principales(donnees_portefeuille, parametres)
        self.graphiques.afficher_graphique_portefeuille(donnees_portefeuille, parametres)
        
        if serie_benchmark is not None and len(serie_benchmark) > 0:
            self.graphiques.afficher_comparaison_benchmark(donnees_portefeuille, serie_benchmark)
        
        self.graphiques.afficher_rendements_mensuels(donnees_portefeuille)
        self.graphiques.afficher_predictions_dca(donnees_prediction)
        
        self.export.afficher_options_export(
            donnees_portefeuille, 
            donnees_prediction, 
            parametres, 
            serie_benchmark
        )

    def executer_simulation(self) -> None:
        """
        Méthode alternative pour l'exécution de simulation (compatibilité).
        """
        if hasattr(st.session_state, 'donnees_portefeuille'):
            self._afficher_resultats(
                st.session_state.donnees_portefeuille,
                st.session_state.serie_benchmark,
                st.session_state.donnees_prediction,
                st.session_state.parametres
            )


if __name__ == "__main__":
    OptinvestApp().run()