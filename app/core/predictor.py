from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson

from app.core.simulator import PortefeuilleSimulator

class PortefeuillePredictorDCA:
    """
    Prédicteur pour les stratégies DCA avec régression linéaire.
    """
    def __init__(self, simulateur: PortefeuilleSimulator | None = None) -> None:
        self.simulateur = simulateur or PortefeuilleSimulator()
        self.frequences_disponibles = ["mensuelle", "trimestrielle", "semestrielle", "annuelle"]

    def predire_avec_regression(self, chronologie_valeurs: pd.Series, duree_prediction_ans: int, volume_serie: pd.Series | None = None) -> Dict:
        """
        Applique deux régressions linéaires : temps seul et temps + volume (si volume fourni).
        """
        donnees_nettoyees = chronologie_valeurs.dropna()
        if len(donnees_nettoyees) < 2:
            raise ValueError("Pas assez de données pour la régression")

        dates_historiques = pd.to_datetime(donnees_nettoyees.index)
        date_reference = dates_historiques.min()
        X_temps = np.array([(d - date_reference).days for d in dates_historiques]).reshape(-1, 1)
        y = np.asarray(donnees_nettoyees.values)

        # Modèle 1 : Temps seul
        modele_temps = LinearRegression()
        modele_temps.fit(X_temps, y)
        valeurs_predites_temps = modele_temps.predict(X_temps)
        residus_temps = y - valeurs_predites_temps

        r2_temps = r2_score(y, valeurs_predites_temps)
        rmse_temps = np.sqrt(mean_squared_error(y, valeurs_predites_temps))
        mae_temps = mean_absolute_error(y, valeurs_predites_temps)
        std_residus_temps = np.std(residus_temps, ddof=1)
        dw_stat_temps = float(durbin_watson(residus_temps))

        # Modèle 2 : Temps + Volume (si volume disponible et aligné)
        if volume_serie is not None:
            volume_serie = volume_serie.reindex(donnees_nettoyees.index).fillna(method="ffill").fillna(method="bfill")
            X_temps_volume = np.column_stack([X_temps.flatten(), volume_serie.values])
            modele_temps_volume = LinearRegression()
            modele_temps_volume.fit(X_temps_volume, y)
            valeurs_predites_temps_volume = modele_temps_volume.predict(X_temps_volume)
            residus_temps_volume = y - valeurs_predites_temps_volume

            r2_temps_volume = r2_score(y, valeurs_predites_temps_volume)
            rmse_temps_volume = np.sqrt(mean_squared_error(y, valeurs_predites_temps_volume))
            mae_temps_volume = mean_absolute_error(y, valeurs_predites_temps_volume)
            std_residus_temps_volume = np.std(residus_temps_volume, ddof=1)
            dw_stat_temps_volume = float(durbin_watson(residus_temps_volume))
            pente_temps = float(modele_temps_volume.coef_[0])
            pente_volume = float(modele_temps_volume.coef_[1])
            intercept_volume = float(modele_temps_volume.intercept_)
        else:
            r2_temps_volume = None
            rmse_temps_volume = None
            mae_temps_volume = None
            std_residus_temps_volume = None
            dw_stat_temps_volume = None
            pente_temps = None
            pente_volume = None
            intercept_volume = None
            residus_temps_volume = None

        date_fin_historique = dates_historiques.max()
        date_fin_prediction = date_fin_historique + timedelta(days=365 * duree_prediction_ans)
        dates_futures = pd.date_range(
            start=date_fin_historique + timedelta(days=30),
            end=date_fin_prediction,
            freq='ME'
        )
        X_futur = np.array([(d - date_reference).days for d in dates_futures]).reshape(-1, 1)
        predictions_futures = modele_temps.predict(X_futur)
        serie_predictions = pd.Series(predictions_futures, index=dates_futures)

        return {
            "predictions": serie_predictions,
            "residus_temps": residus_temps.tolist(),
            "residus_temps_volume": residus_temps_volume.tolist() if residus_temps_volume is not None else None,
            "regression_metrics_temps": {
                "r2": float(r2_temps),
                "rmse": float(rmse_temps),
                "mae": float(mae_temps),
                "std_residus": float(std_residus_temps),
                "pente": float(modele_temps.coef_[0]),
                "intercept": float(modele_temps.intercept_),
                "dw_stat": dw_stat_temps
            },
            "regression_metrics_temps_volume": {
                "r2": r2_temps_volume,
                "rmse": rmse_temps_volume,
                "mae": mae_temps_volume,
                "std_residus": std_residus_temps_volume,
                "pente_temps": pente_temps,
                "pente_volume": pente_volume,
                "intercept": intercept_volume,
                "dw_stat": dw_stat_temps_volume
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
        Compare différentes stratégies DCA et lump sum avec prédictions (temps seul et temps+volume).
        """
        resultats_strategies = {}

        duree_simulation_ans = (end - start).days / 365
        nombre_apports_mensuels = int(duree_simulation_ans * 12)
        montant_total_disponible = montant_initial + (apport_periodique * nombre_apports_mensuels)

        try:
            donnees_volume = self.simulateur.extracteur.batch_fetch_volume(actifs, start, end)
            volume_moyen = donnees_volume.mean(axis=1)
        except Exception:
            volume_moyen = None

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

                resultats_prediction = self.predire_avec_regression(
                    chronologie_historique, 
                    duree_prediction_ans, 
                    volume_serie=volume_moyen
                )

                predictions_formatees = {
                    date_prediction.strftime("%Y-%m-%d"): float(valeur_predite)
                    for date_prediction, valeur_predite in resultats_prediction["predictions"].items()
                }

                resultats_strategies[f"DCA {frequence_courante.title()} (Temps seul)"] = {
                    "timeline_historique": resultats_simulation["timeline"],
                    "predictions": predictions_formatees,
                    "regression_metrics": resultats_prediction["regression_metrics_temps"],
                    "residus": resultats_prediction["residus_temps"],
                    "cagr": resultats_simulation["cagr"],
                    "rendement_total": resultats_simulation["rendement_total"]
                }

                if resultats_prediction["regression_metrics_temps_volume"]["r2"] is not None:
                    resultats_strategies[f"DCA {frequence_courante.title()} (Temps+Volume)"] = {
                        "timeline_historique": resultats_simulation["timeline"],
                        "predictions": predictions_formatees,
                        "regression_metrics": resultats_prediction["regression_metrics_temps_volume"],
                        "residus": resultats_prediction["residus_temps_volume"],
                        "cagr": resultats_simulation["cagr"],
                        "rendement_total": resultats_simulation["rendement_total"]
                    }

            except Exception as erreur_dca:
                print(f"Erreur pour DCA {frequence_courante}: {erreur_dca}")
                continue

        # Lump Sum
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

            predictions_lump_sum = self.predire_avec_regression(
                chronologie_lump_sum, 
                duree_prediction_ans, 
                volume_serie=volume_moyen
            )

            predictions_lump_formatees = {
                date_prediction.strftime("%Y-%m-%d"): float(valeur_predite)
                for date_prediction, valeur_predite in predictions_lump_sum["predictions"].items()
            }

            resultats_strategies["Lump Sum (Temps seul)"] = {
                "timeline_historique": resultats_lump_sum["timeline"],
                "predictions": predictions_lump_formatees,
                "regression_metrics": predictions_lump_sum["regression_metrics_temps"],
                "residus": predictions_lump_sum["residus_temps"],
                "cagr": resultats_lump_sum["cagr"],
                "rendement_total": resultats_lump_sum["rendement_total"]
            }

            if predictions_lump_sum["regression_metrics_temps_volume"]["r2"] is not None:
                resultats_strategies["Lump Sum (Temps+Volume)"] = {
                    "timeline_historique": resultats_lump_sum["timeline"],
                    "predictions": predictions_lump_formatees,
                    "regression_metrics": predictions_lump_sum["regression_metrics_temps_volume"],
                    "residus": predictions_lump_sum["residus_temps_volume"],
                    "cagr": resultats_lump_sum["cagr"],
                    "rendement_total": resultats_lump_sum["rendement_total"]
                }

        except Exception as erreur_lump_sum:
            print(f"Erreur pour Lump Sum: {erreur_lump_sum}")

        return {"strategies": resultats_strategies}