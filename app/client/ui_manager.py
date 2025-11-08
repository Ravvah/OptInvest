from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional

import streamlit as st

class InterfaceUtilisateur:
    ACTIFS_AUTORISES: List[str] = [
        "AAPL", "MSFT", "AGGH", "TLT", "VWCE.DE", "SXR8.DE"
    ]
    
    FREQUENCES: Dict[str, str] = {
        "Mensuelle": "mensuelle",
        "Trimestrielle": "trimestrielle", 
        "Semestrielle": "semestrielle",
        "Annuelle": "annuelle",
    }
    
    MOIS_PAR_FREQ: Dict[str, int] = {
        "mensuelle": 1,
        "trimestrielle": 3,
        "semestrielle": 6,
        "annuelle": 12,
    }

    def afficher_sidebar(self) -> Optional[Dict]:
        """
        Affiche la barre latérale et retourne les paramètres de simulation.
        """
        with st.sidebar:
            st.header("Paramètres de la simulation")
            
            actifs_selectionnes = st.multiselect(
                "Choisissez vos actifs", 
                self.ACTIFS_AUTORISES, 
                default=["AAPL"]
            )
            
            duree_simulation_annees = st.slider("Durée (années)", 1, 20, 6)
            montant_initial = st.number_input("Montant initial (€)", 1.0, 1_000_000.0, 10_000.0, 100.0)
            apport_periodique = st.number_input("Apport périodique (€)", 0.0, 100_000.0, 200.0, 50.0)
            
            frequence_affichage = st.selectbox(
                "Fréquence d'investissement", 
                list(self.FREQUENCES.keys()), 
                0
            )
            frequence_investissement = self.FREQUENCES[frequence_affichage]
            
            frais_gestion = st.number_input("Frais de gestion annuels (%)", 0.0, 5.0, 0.2, 0.1)
            
            st.header("Prédictions")
            duree_prediction_annees = st.slider("Durée de prédiction (années)", 1, 10, 5)
            
            bouton_lancer = st.button("Simuler")

        if bouton_lancer:
            if not actifs_selectionnes:
                st.error("Veuillez sélectionner au moins un actif.")
                return None

            return self._construire_parametres(
                actifs_selectionnes,
                duree_simulation_annees,
                montant_initial,
                apport_periodique,
                frequence_investissement,
                frais_gestion,
                duree_prediction_annees
            )
        
        return None

    def _construire_parametres(
        self,
        actifs_selectionnes: List[str],
        duree_simulation_annees: int,
        montant_initial: float,
        apport_periodique: float,
        frequence_investissement: str,
        frais_gestion: float,
        duree_prediction_annees: int
    ) -> Dict:
        """
        Construit le dictionnaire des paramètres de simulation.
        """

        # Définir la date de début par défaut
        date_debut = str(date.today() - timedelta(days=365 * duree_simulation_annees))

        parametres_base = {
            "actifs": actifs_selectionnes,
            "duree_ans": duree_simulation_annees,
            "date_debut": date_debut,
            "montant_initial": montant_initial,
            "apport_periodique": apport_periodique,
            "frequence": frequence_investissement,
            "frais_gestion": frais_gestion,
        }

        parametres_prediction = {
            **parametres_base,
            "duree_prediction_ans": duree_prediction_annees
        }

        return {
            "base": parametres_base,
            "prediction": parametres_prediction,
            "duree_simulation_annees": duree_simulation_annees,
            "frequence_investissement": frequence_investissement,
            "montant_initial": montant_initial,
            "apport_periodique": apport_periodique,
            "duree_prediction_annees": duree_prediction_annees
        }

    def calculer_cash_investi(
        self, 
        duree_simulation_annees: int, 
        frequence_investissement: str, 
        montant_initial: float, 
        apport_periodique: float
    ) -> float:
        """
        Calcule le montant total de cash investi.
        """
        nombre_periodes = (duree_simulation_annees * 12) // self.MOIS_PAR_FREQ[frequence_investissement]
        return montant_initial + nombre_periodes * apport_periodique