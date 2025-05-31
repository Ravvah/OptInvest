# app/client/main.py
"""Interface Streamlit orientée POO pour OPTINVEST.

Lancez avec :
    streamlit run app/client/main.py
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st


class OptinvestApp:
    """Application Streamlit"""

    TICKERS_AUTORISES: List[str] = [
        "AAPL",  # Action
        "MSFT",  # Action
        "AGGH",  # ETF obligataire global
        "TLT",   # ETF obligations US long terme
        "VWCE",  # ETF actions monde
        "SXR8",  # ETF S&P 500
    ]

    FREQUENCES: Dict[str, str] = {
        "Mensuelle": "mensuelle",
        "Trimestrielle": "trimestrielle",
        "Semestrielle": "semestrielle",
        "Annuelle": "annuelle",
    }

    API_URL = "http://localhost:8000/api/simuler"

    def _afficher_resultats(self, data: Dict[str, float]) -> None:
        """Affiche métriques + graphique + export CSV."""
        st.success("Simulation terminée !")

        col1, col2 = st.columns(2)
        col1.metric("CAGR", f"{data['cagr']*100:.2f} %")
        col2.metric("Rendement total", f"{data['rendement_total']:.2f} €")

        timeline_series = pd.Series(data["timeline"], name="Valeur €", dtype=float)
        timeline_series.index = pd.to_datetime(timeline_series.index)
        st.line_chart(timeline_series)

        st.dataframe(timeline_series.to_frame())
        st.download_button(
            "📥 Télécharger CSV",
            data=timeline_series.to_csv().encode("utf-8"),
            file_name="simulation_optinvest.csv",
            mime="text/csv",
        )

    def _appeler_api(self, payload: Dict) -> Dict:
        """Envoie la requête POST et renvoie le JSON."""
        resp = requests.post(self.API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # Interface UI
    def run(self) -> None:
        """Point d’entrée : construit l’UI et gère la logique."""

        st.set_page_config(
            page_title="OPTINVEST – Simulation de portefeuille",
            page_icon="📈",
            layout="centered",
        )
        st.title("📈 OPTINVEST – Simulation de portefeuille")

        # ---------- Barre latérale ----------
        with st.sidebar:
            st.header("Paramètres de la simulation")

            actifs = st.multiselect(
                "Choisissez vos actifs",
                options=self.TICKERS_AUTORISES,
                default=["VWCE"],
            )

            duree_ans = st.slider("Durée (années)", min_value=1, max_value=20, value=6)

            montant_initial = st.number_input(
                "Montant initial (€)",
                min_value=1.0,
                value=10_000.0,
                step=100.0,
                format="%f",
            )

            apport = st.number_input(
                "Apport périodique (€)",
                min_value=0.0,
                value=200.0,
                step=50.0,
                format="%f",
            )

            frequence_humaine = st.selectbox(
                label="Fréquence d'investissement",
                options=list(self.FREQUENCES.keys()),
                index=0
            )
            frequence = self.FREQUENCES[frequence_humaine]

            frais = st.number_input(
                "Frais de gestion annuels (%)",
                min_value=0.0,
                value=0.2,
                step=0.1,
                format="%f",
            )

            lancer = st.button("▶️ Simuler")

        if lancer:
            if not actifs:
                st.error("Veuillez sélectionner au moins un actif.")
                st.stop()

            payload = {
                "actifs": actifs,
                "duree_ans": duree_ans,
                "montant_initial": montant_initial,
                "apport_periodique": apport,
                "frequence": frequence,
                "frais_gestion": frais,
            }

            with st.spinner("Calcul en cours…"):
                try:
                    data = self._appeler_api(payload)
                except requests.exceptions.RequestException as err:
                    st.error(f"Erreur lors de l'appel API : {err}")
                    st.stop()

            self._afficher_resultats(data)


if __name__ == "__main__":
    OptinvestApp().run()
