from datetime import date
from typing import Dict, List
import pandas as pd
from app.core.datasources.extract_yfinance import YahooExtractor

FREQ2MONTH = {"mensuelle": 1, "trimestrielle": 3, "semestrielle": 6, "annuelle": 12}

class PortefeuilleSimulator:
    """
    Simule un portefeuille passif (pondération 1/n).
    """

    def __init__(self, extractor: YahooExtractor | None = None) -> None:
        self.extracteur = extractor or YahooExtractor()

    def _charger_prix(self, actifs: List[str], start: date, end: date) -> pd.Series:
        donnees_prix = self.extracteur.batch_fetch(actifs, start, end)
        return donnees_prix.mean(axis=1)

    def simuler(
        self,
        actifs: List[str],
        start: date,
        end: date,
        montant_initial: float,
        apport_periodique: float,
        frequence: str,
        frais_gestion_pct: float,
    ) -> Dict:
        """
        Simule l'évolution d'un portefeuille avec apports périodiques.
        """
        prix_moyen = self._charger_prix(actifs, start, end)

        nombre_mois = FREQ2MONTH[frequence]
        dates_apport = pd.date_range(start=start, end=end, freq=f"{nombre_mois}ME").date

        nombre_parts = 0.0
        cash_investi_total = 0.0
        chronologie_valeurs: Dict[str, float] = {}
        
        montant_initial_investi = False

        for date_courante, prix_courant in prix_moyen.items():
            date_courante_convertie = pd.to_datetime(str(date_courante)).date()

            if not montant_initial_investi:
                nombre_parts += montant_initial / prix_courant
                cash_investi_total += montant_initial
                montant_initial_investi = True

            elif date_courante_convertie in dates_apport:
                nombre_parts += apport_periodique / prix_courant
                cash_investi_total += apport_periodique

            # Frais de gestion annuels
            if date_courante_convertie.month == 12 and date_courante_convertie.day == 31:
                nombre_parts *= 1 - frais_gestion_pct / 100

            chronologie_valeurs[str(date_courante_convertie)] = nombre_parts * prix_courant

        dates_valides = [
            pd.to_datetime(k).date() for k in chronologie_valeurs.keys() 
            if pd.to_datetime(k).date() <= end
        ]
        
        if dates_valides:
            date_fin_effective = max(dates_valides)
        else:
            date_fin_effective = min(pd.to_datetime(k).date() for k in chronologie_valeurs.keys())
        
        valeur_finale_portefeuille = chronologie_valeurs[str(date_fin_effective)]

        duree_simulation_annees = (date_fin_effective - start).days / 365
        taux_croissance_annuel = (
            (valeur_finale_portefeuille / cash_investi_total) ** (1 / duree_simulation_annees) - 1 
            if duree_simulation_annees > 0 else 0.0
        )
        rendement_net_total = valeur_finale_portefeuille - cash_investi_total

        return {
            "cagr": round(taux_croissance_annuel, 4),
            "rendement_total": round(rendement_net_total, 2),
            "timeline": chronologie_valeurs,
        }