from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

from app.core.simulator import PortefeuilleSimulator

class PortefeuillePredictorDCA:
    """
    Prédicteur pour les stratégies DCA avec régression linéaire.
    """
    
    def __init__(self, simulateur: PortefeuilleSimulator | None = None) -> None:
        self.simulateur = simulateur or PortefeuilleSimulator()
        self.frequences_disponibles = ["mensuelle", "trimestrielle", "semestrielle", "annuelle"]
    
    def predire_avec_regression(self, chronologie_valeurs: pd.Series, duree_prediction_ans: int) -> Dict:
        """
        Applique une régression linéaire pour prédire les rendements futurs.
        """
        donnees_nettoyees = chronologie_valeurs.dropna()
        if len(donnees_nettoyees) < 2:
            raise ValueError("Pas assez de données pour la régression")
        
        dates_historiques = pd.to_datetime(donnees_nettoyees.index)
        
        date_reference = dates_historiques.min()
        X = np.array([(d - date_reference).days for d in dates_historiques]).reshape(-1, 1)
        y = np.asarray(donnees_nettoyees.values)
        
        modele_regression = LinearRegression()
        modele_regression.fit(X, y)
        
        valeurs_predites = modele_regression.predict(X)
        residus = y - valeurs_predites
        
        coefficient_determination = r2_score(y, valeurs_predites)
        erreur_quadratique_moyenne = np.sqrt(mean_squared_error(y, valeurs_predites))
        erreur_absolue_moyenne = mean_absolute_error(y, valeurs_predites)
        
        ecart_type_residus = np.std(residus, ddof=1)  # ddof=1 pour échantillon
                
        date_fin_historique = dates_historiques.max()
        date_fin_prediction = date_fin_historique + timedelta(days=365 * duree_prediction_ans)
        
        dates_futures = pd.date_range(
            start=date_fin_historique + timedelta(days=30),
            end=date_fin_prediction,
            freq='ME'
        )
        
        X_futur = np.array([(d - date_reference).days for d in dates_futures]).reshape(-1, 1)
        predictions_futures = modele_regression.predict(X_futur)
        
        serie_predictions = pd.Series(predictions_futures, index=dates_futures)
        
        return {
            "predictions": serie_predictions,
            "residus": residus.tolist(),
            "regression_metrics": {
                "r2": float(coefficient_determination),
                "rmse": float(erreur_quadratique_moyenne),
                "mae": float(erreur_absolue_moyenne),
                "std_residus": float(ecart_type_residus),
                "pente": float(modele_regression.coef_[0]),
                "intercept": float(modele_regression.intercept_),
            }
        }
    
    def comparer_strategies_dca(
        self,
        actifs: List[str],
        start: date,
        end: date,
        montant_initial: float,
        apport_periodique: float,
        frais_gestion_pct: float,
        duree_prediction_ans: int
    ) -> Dict:
        """
        Compare différentes stratégies DCA et lump sum avec prédictions.
        """
        resultats_strategies = {}
        
        duree_simulation_ans = (end - start).days / 365
        
        nombre_apports_mensuels = int(duree_simulation_ans * 12)
        montant_total_disponible = montant_initial + (apport_periodique * nombre_apports_mensuels)
        
        for frequence_courante in self.frequences_disponibles:
            try:
                resultats_simulation = self.simulateur.simuler(
                    actifs=actifs,
                    start=start,
                    end=end,
                    montant_initial=montant_initial,
                    apport_periodique=apport_periodique,
                    frequence=frequence_courante,
                    frais_gestion_pct=frais_gestion_pct
                )
                
                chronologie_historique = pd.Series(resultats_simulation["timeline"])
                chronologie_historique.index = pd.to_datetime(chronologie_historique.index)
                
                resultats_prediction = self.predire_avec_regression(chronologie_historique, duree_prediction_ans)
                
                predictions_formatees = {
                    str(date_prediction): float(valeur_predite) 
                    for date_prediction, valeur_predite in resultats_prediction["predictions"].items()
                }
                
                resultats_strategies[f"DCA {frequence_courante.title()}"] = {
                    "timeline_historique": resultats_simulation["timeline"],
                    "predictions": predictions_formatees,
                    "regression_metrics": resultats_prediction["regression_metrics"],
                    "residus": resultats_prediction["residus"],
                    "cagr": resultats_simulation["cagr"],
                    "rendement_total": resultats_simulation["rendement_total"]
                }
                
            except Exception as erreur_dca:
                print(f"Erreur pour DCA {frequence_courante}: {erreur_dca}")
                continue
        
        try:
            resultats_lump_sum = self.simulateur.simuler(
                actifs=actifs,
                start=start,
                end=end,
                montant_initial=montant_total_disponible,
                apport_periodique=0.0,
                frequence="mensuelle",
                frais_gestion_pct=frais_gestion_pct
            )
            
            chronologie_lump_sum = pd.Series(resultats_lump_sum["timeline"])
            chronologie_lump_sum.index = pd.to_datetime(chronologie_lump_sum.index)
            
            predictions_lump_sum = self.predire_avec_regression(chronologie_lump_sum, duree_prediction_ans)
            
            predictions_lump_formatees = {
                str(date_prediction): float(valeur_predite) 
                for date_prediction, valeur_predite in predictions_lump_sum["predictions"].items()
            }
            
            resultats_strategies["Lump Sum"] = {
                "timeline_historique": resultats_lump_sum["timeline"],
                "predictions": predictions_lump_formatees,
                "regression_metrics": predictions_lump_sum["regression_metrics"],
                "residus": predictions_lump_sum["residus"],
                "cagr": resultats_lump_sum["cagr"],
                "rendement_total": resultats_lump_sum["rendement_total"]
            }
            
        except Exception as erreur_lump_sum:
            print(f"Erreur pour Lump Sum: {erreur_lump_sum}")
        
        return {"strategies": resultats_strategies}