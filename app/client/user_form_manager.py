from __future__ import annotations

from typing import Any, Dict
import streamlit as st

from app.api.schemas.prediction import PredictionRequest
from app.api.schemas.simulation import SimulationRequest
import app.client.constants as cst

class UserForm:

    def __init__(self):
        pass

    def display_user_form_sidebar(self) -> Dict[str, Any]:
        """
        Affiche la barre latérale et retourne les paramètres de simulation.
        """
        with st.sidebar:
            st.header("Paramètres de la simulation")
            
            actifs_selectionnes = st.multiselect(
                "Choisissez vos actifs", 
                cst.ACTIFS_AUTORISES, 
                default=["AAPL"]
            )
            
            duree_simulation_annees = st.slider("Durée (années)", 1, 20, 6)
            montant_initial = st.number_input("Montant initial (€)", 1.0, 1_000_000.0, 10_000.0, 100.0)
            apport_periodique = st.number_input("Apport périodique (€)", 0.0, 100_000.0, 200.0, 50.0)
            
            frequence_affichage = st.selectbox(
                "Fréquence d'investissement", 
                list(cst.FREQUENCES.keys()), 
                0
            )
            frequence_investissement = cst.FREQUENCES[frequence_affichage]
            
            frais_gestion = st.number_input("Frais de gestion annuels (%)", 0.0, 5.0, 0.2, 0.1)
            
            st.header("Prédictions")
            duree_prediction_ans = st.slider("Durée de prédiction (années)", 1, 10, 5)
            
            bouton_lancer = st.button("Simuler")
        
        return dict(actifs_selectionnes=actifs_selectionnes,
                    duree_simulation_annees=duree_simulation_annees,
                    montant_initial=montant_initial,
                    apport_periodique=apport_periodique,
                    frequence_investissement=frequence_investissement,
                    frais_gestion=frais_gestion,
                    duree_prediction_ans=duree_prediction_ans,
                    lancer=bouton_lancer) if bouton_lancer else None
                    

    def _build_portfolio_simulation_request(
        self,
        inputs: dict
    ) -> tuple[SimulationRequest, str]:
        """
        Construit le dictionnaire des paramètres de simulation.
        """

        if not inputs:
            raise ValueError("Aucun paramètre d'entrée fourni pour la simulation.")
        
        return SimulationRequest(
            actifs=inputs["actifs_selectionnes"],
            duree_ans=inputs["duree_simulation_annees"],
            montant_initial=inputs["montant_initial"],
            apport_periodique=inputs["apport_periodique"],
            frequence=inputs["frequence_investissement"],
            frais_gestion=inputs["frais_gestion"],
            
        ), inputs["duree_prediction_ans"]
    
    def _build_prediction_request(
        self,
        valeur_portefeuille_temps: dict[str, float],
        valeur_portefeuille_montant: dict[str, float],
        duree_prediction_ans: int,
        apport_periodique: float,
        frequence: str,
        modele: str

    ) -> PredictionRequest:
        """
        Construit le dictionnaire des paramètres de prédiction.
        """
        if not valeur_portefeuille_temps:
            raise ValueError("Pas de timeline trouvée pour construire ")
        
        date_derniere_simulation = list(valeur_portefeuille_temps.keys())[-1]
        liste_montants = [[float(key)] for key in valeur_portefeuille_montant.keys()]
        liste_valeurs_portefeuille = list(valeur_portefeuille_montant.values())

        return PredictionRequest(
            x=liste_montants,
            y=liste_valeurs_portefeuille,
            duree_prediction_ans=duree_prediction_ans,
            apport_periodique=apport_periodique,
            frequence=frequence,
            date_derniere_simulation=date_derniere_simulation,
            modele=modele
        )