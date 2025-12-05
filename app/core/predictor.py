from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean, variance
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

from app.api.schemas.constants import FREQ2MONTH
from app.api.schemas.prediction import CoefficientAnalysis, FitQualityAnalysis, OverfittingAnalysis, PredictionRequest, PredictionResponse, ResidualAnalysis

class Predictor(ABC):
    def __init__(self, requete: PredictionRequest):
        self.requete = requete
    
    @abstractmethod
    def _fit(self):
        pass

    @abstractmethod
    def _predict(self, x_new: List[float]) -> float:
        pass

    def _generate_new_inputs(self) -> tuple[List[List[float]], List[str]]:
        if self.requete.frequence is None:
            nombre_mois_prediction = 1 
            apport_futur = 0.0
            nb_periodes = self.requete.duree_prediction_ans * 12
        else:
            nombre_mois_prediction = FREQ2MONTH[self.requete.frequence]
            apport_futur = self.requete.apport_periodique
            nb_periodes = (self.requete.duree_prediction_ans * 12) // nombre_mois_prediction
            
        
        date_debut_prediction = self.requete.date_derniere_simulation + pd.DateOffset(months= nombre_mois_prediction)
        new_dates = [date_debut_prediction + pd.DateOffset(months= i * nombre_mois_prediction) for i in range(nb_periodes)]
        last_existing_value = self.requete.x[-1][0]
        
        new_inputs = [[last_existing_value + apport_futur * i] for i in range(nb_periodes)]
        
        str_new_dates = [str(new_date_i.date()) for new_date_i in new_dates]
        return new_inputs, str_new_dates


class LinearModelPredictor(Predictor):
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None
    
    def _fit(self):
        X = sm.add_constant(self.requete.x)
        self.model = sm.OLS(self.requete.y, X).fit()

    def _predict(self, x_new: List[List[float]]) -> List[float]:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        X_new = np.array(x_new)
        X_new = sm.add_constant(X_new, has_constant="add")
        y_new = self.model.predict(X_new)
        return y_new
    
    
    def _get_r_squared(self) -> float:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")

        return self.model.rsquared  
    
    def _get_rmse(self) -> float:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        residuals = self.model.resid.tolist()
        rmse = np.sqrt(np.mean([ri **2 for ri in residuals]))
        y_mean = np.mean(self.requete.y)
        rmse_relatif = (rmse / y_mean) * 100
        return rmse_relatif
    
    
    def _get_coeff_p_values(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        return {coeff : pvalue  for coeff, pvalue in zip(self.model.model.exog_names, self.model.pvalues)}
    
    def _get_coeff_confidence_intervals(self) -> Dict[str, List[float]]:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")

        return {coeff: list(interval) for coeff, interval in zip(self.model.model.exog_names, self.model.conf_int())}
    
    def _get_residual_analysis(self) -> ResidualAnalysis:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        residuals = self.model.resid
        autocorrelation_stat = durbin_watson(residuals)
        return ResidualAnalysis(
            residuals=residuals.tolist(),
            moyenne=np.mean(residuals),
            ecart_type=np.sqrt(np.var(residuals, ddof=1)),
            autocorrelation_statistic=autocorrelation_stat
        )
    
    def _get_fit_quality_analysis(self) -> FitQualityAnalysis:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        return FitQualityAnalysis(
            r2=self._get_r_squared(),
            rmse=self._get_rmse()
        )
    
    def _get_coeff_analysis(self) -> CoefficientAnalysis:
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été ajusté !")
        
        return CoefficientAnalysis(
            coeff_p_values=self._get_coeff_p_values(),
            coeff_confidence_intervals=self._get_coeff_confidence_intervals()
        )
    
    def _get_overfitting_analysis(self) -> OverfittingAnalysis:
        total_samples = len(self.requete.y)

        if total_samples < 4:
            return OverfittingAnalysis(r2_diff=0.0)   # valeur neutre

        test_size = max(2, int(total_samples * 0.2))
        if test_size >= total_samples - 2:
            test_size = 2

        x_train = self.requete.x[:-test_size]
        y_train = self.requete.y[:-test_size]
        x_val = self.requete.x[-test_size:]
        y_val = self.requete.y[-test_size:]

        if len(y_train) < 2 or len(y_val) < 2:
            return OverfittingAnalysis(r2_diff=0.0)

        X_train = sm.add_constant(x_train, has_constant="add")
        model_temp = sm.OLS(y_train, X_train).fit()

        X_val = sm.add_constant(x_val, has_constant="add")
        y_pred_val = model_temp.predict(X_val)

        # R² train
        r2_train = model_temp.rsquared
        if r2_train is None or np.isnan(r2_train):
            r2_train = 0.0

        # R² val
        try:
            r2_val = r2_score(y_val, y_pred_val)
            if np.isnan(r2_val):
                r2_val = 0.0
        except Exception:
            r2_val = 0.0

        r2_diff = abs(r2_train - r2_val)

        return OverfittingAnalysis(r2_diff=float(r2_diff))


    def _get_prediction_response(self) -> PredictionResponse:
        if self.model is None:
            self._fit()

        new_inputs = self._generate_new_inputs()
        new_outputs = self._predict(new_inputs[0])
        new_predictions = {str(xi[0]) : y for xi, y in zip(new_inputs[0], new_outputs)}
        new_predictions_timeline = {date_i : y for date_i, y in zip(new_inputs[1], new_outputs)}
        fit_quality_analysis = self._get_fit_quality_analysis()
        coeff_analysis = self._get_coeff_analysis()
        residual_analysis = self._get_residual_analysis()
        overfitting_analysis = self._get_overfitting_analysis()
        resultats = PredictionResponse(
            model=self.requete.modele,
            predictions=new_predictions,
            predictions_timeline=new_predictions_timeline,
            fit_quality_analysis=fit_quality_analysis,
            coeff_analysis=coeff_analysis,
            residual_analysis=residual_analysis,
            overfitting_analysis=overfitting_analysis
        )
            
        return resultats

#TO DO: implement an AR Processus as the second model 
class AutoRegressiveModelPredictor(Predictor):
    """
    Class for AutoRegressive Model Predictor"""
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None
    


#TO DO: implement an ARMA Processus as the third model
class ARMA1ModelPredictor(Predictor):
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None


#TO DO: implement an ARMA-GARCH Processus as the fourth model
class ARMAGARCH1ModelPredictor(Predictor):
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None        
        





    
   








# class PortefeuillePredictorDCA:
#     """
#     Prédicteur pour les stratégies DCA avec régression linéaire.
#     """
#     def __init__(self, simulateur: PortfolioSimulator | None = None) -> None:
#         self.simulateur = simulateur or PortfolioSimulator()
#         self.frequences_disponibles = ["mensuelle", "trimestrielle", "semestrielle", "annuelle"]

#     def predire_avec_regression(self, chronologie_valeurs: pd.Series, duree_prediction_ans: int, volume_serie: pd.Series | None = None) -> Dict:
#         """
#         Applique deux régressions linéaires : temps seul et temps + volume (si volume fourni).
#         """
#         donnees_nettoyees = chronologie_valeurs.dropna()
#         if len(donnees_nettoyees) < 2:
#             raise ValueError("Pas assez de données pour la régression")

#         dates_historiques = pd.to_datetime(donnees_nettoyees.index)
#         date_reference = dates_historiques.min()
#         X_temps = np.array([(d - date_reference).days for d in dates_historiques]).reshape(-1, 1)
#         y = np.asarray(donnees_nettoyees.values)

#         # Modèle 1 : Temps seul
#         modele_temps = LinearRegression()
#         modele_temps.fit(X_temps, y)
#         valeurs_predites_temps = modele_temps.predict(X_temps)
#         residus_temps = y - valeurs_predites_temps

#         r2_temps = r2_score(y, valeurs_predites_temps)
#         rmse_temps = np.sqrt(mean_squared_error(y, valeurs_predites_temps))
#         mae_temps = mean_absolute_error(y, valeurs_predites_temps)
#         std_residus_temps = np.std(residus_temps, ddof=1)
#         dw_stat_temps = float(durbin_watson(residus_temps))

#         # Modèle 2 : Temps + Volume (si volume disponible et aligné)
#         if volume_serie is not None:
#             volume_serie = volume_serie.reindex(donnees_nettoyees.index).fillna(method="ffill").fillna(method="bfill")
#             X_temps_volume = np.column_stack([X_temps.flatten(), volume_serie.values])
#             modele_temps_volume = LinearRegression()
#             modele_temps_volume.fit(X_temps_volume, y)
#             valeurs_predites_temps_volume = modele_temps_volume.predict(X_temps_volume)
#             residus_temps_volume = y - valeurs_predites_temps_volume

#             r2_temps_volume = r2_score(y, valeurs_predites_temps_volume)
#             rmse_temps_volume = np.sqrt(mean_squared_error(y, valeurs_predites_temps_volume))
#             mae_temps_volume = mean_absolute_error(y, valeurs_predites_temps_volume)
#             std_residus_temps_volume = np.std(residus_temps_volume, ddof=1)
#             dw_stat_temps_volume = float(durbin_watson(residus_temps_volume))
#             pente_temps = float(modele_temps_volume.coef_[0])
#             pente_volume = float(modele_temps_volume.coef_[1])
#             intercept_volume = float(modele_temps_volume.intercept_)
#         else:
#             r2_temps_volume = None
#             rmse_temps_volume = None
#             mae_temps_volume = None
#             std_residus_temps_volume = None
#             dw_stat_temps_volume = None
#             pente_temps = None
#             pente_volume = None
#             intercept_volume = None
#             residus_temps_volume = None

#         date_fin_historique = dates_historiques.max()
#         date_fin_prediction = date_fin_historique + timedelta(days=365 * duree_prediction_ans)
#         dates_futures = pd.date_range(
#             start=date_fin_historique + timedelta(days=30),
#             end=date_fin_prediction,
#             freq='ME'
#         )
#         X_futur = np.array([(d - date_reference).days for d in dates_futures]).reshape(-1, 1)
#         predictions_futures = modele_temps.predict(X_futur)
#         serie_predictions = pd.Series(predictions_futures, index=dates_futures)

#         return {
#             "predictions": serie_predictions,
#             "residus_temps": residus_temps.tolist(),
#             "residus_temps_volume": residus_temps_volume.tolist() if residus_temps_volume is not None else None,
#             "regression_metrics_temps": {
#                 "r2": float(r2_temps),
#                 "rmse": float(rmse_temps),
#                 "mae": float(mae_temps),
#                 "std_residus": float(std_residus_temps),
#                 "pente": float(modele_temps.coef_[0]),
#                 "intercept": float(modele_temps.intercept_),
#                 "dw_stat": dw_stat_temps
#             },
#             "regression_metrics_temps_volume": {
#                 "r2": r2_temps_volume,
#                 "rmse": rmse_temps_volume,
#                 "mae": mae_temps_volume,
#                 "std_residus": std_residus_temps_volume,
#                 "pente_temps": pente_temps,
#                 "pente_volume": pente_volume,
#                 "intercept": intercept_volume,
#                 "dw_stat": dw_stat_temps_volume
#             }
#         }
    
#     def predire_avec_processus_ar(self, chronologie_valeurs: pd.Series, lags: int = 1, horizon: int = 12) -> Dict:
#         """
#         Ajuste un modèle AR(p) sur les valeurs brutes du portefeuille et prédit sur l'horizon demandé.
#         """
#         serie = chronologie_valeurs.dropna().sort_index()
#         if len(serie) < lags + 2:
#             raise ValueError("Pas assez de données pour le modèle AR")

#         # AR sur valeurs brutes
#         modele_ar = AutoReg(serie, lags=lags, old_names=False).fit()
#         params = modele_ar.params.to_dict()
#         aic = modele_ar.aic
#         bic = modele_ar.bic
#         residus = modele_ar.resid

#         # Prévision sur horizon - UTILISER forecast() au lieu de predict()
#         forecast = modele_ar.forecast(steps=horizon)
        
#         # Créer les dates futures
#         dates_predites = pd.date_range(serie.index[-1], periods=horizon+1, freq="ME")[1:]
        
#         # IMPORTANT: utiliser .values pour éviter les problèmes d'alignement d'index
#         predictions = pd.Series(forecast.values, index=dates_predites)
#         print(f"DEBUG AR: Prédictions = {predictions.values[:5]}...")  # 5 premières valeurs

#         # Convertir les clés Timestamp en chaînes de caractères pour Pydantic
#         predictions_formatees = {
#             date_pred.strftime("%Y-%m-%d"): float(valeur)
#             for date_pred, valeur in predictions.items()
#         }

#         return {
#             "params": params,
#             "aic": float(aic),
#             "bic": float(bic),
#             "residus": residus.tolist(),
#             "predictions": predictions_formatees
#         }
    
#     def comparer_strategies_dca(
#         self,
#         actifs: List[str],
#         start: date,
#         end: date,
#         montant_initial: float,
#         apport_periodique: float,
#         frais_gestion_pct: float,
#         duree_prediction_ans: int
#     ) -> Dict:
#         """
#         Compare différentes stratégies DCA et lump sum avec prédictions (temps seul et temps+volume).
#         """
#         resultats_strategies = {}

#         duree_simulation_ans = (end - start).days / 365
#         nombre_apports_mensuels = int(duree_simulation_ans * 12)
#         montant_total_disponible = montant_initial + (apport_periodique * nombre_apports_mensuels)

#         try:
#             donnees_volume = self.simulateur.extracteur.concat_multiple_volume(actifs, start, end)
#             volume_moyen = donnees_volume.mean(axis=1)
#         except Exception:
#             volume_moyen = None

#         for frequence_courante in self.frequences_disponibles:
#             try:
#                 resultats_simulation = self.simulateur.simulate_portfolio_growth(
#                     actifs=actifs,
#                     start=start,
#                     end=end,
#                     montant_initial=montant_initial,
#                     apport_periodique=apport_periodique,
#                     frequence=frequence_courante,
#                     frais_gestion_pct=frais_gestion_pct
#                 )

#                 chronologie_historique = pd.Series(resultats_simulation["timeline"])
#                 chronologie_historique.index = pd.to_datetime(chronologie_historique.index)

#                 resultats_prediction = self.predire_avec_regression(
#                     chronologie_historique, 
#                     duree_prediction_ans, 
#                     volume_serie=volume_moyen
#                 )

#                 predictions_formatees = {
#                     date_prediction.strftime("%Y-%m-%d"): float(valeur_predite)
#                     for date_prediction, valeur_predite in resultats_prediction["predictions"].items()
#                 }

#                 resultats_strategies[f"DCA {frequence_courante.title()} (Temps seul)"] = {
#                     "timeline_historique": resultats_simulation["timeline"],
#                     "predictions": predictions_formatees,
#                     "regression_metrics": resultats_prediction["regression_metrics_temps"],
#                     "residus": resultats_prediction["residus_temps"],
#                     "cagr": resultats_simulation["cagr"],
#                     "rendement_total": resultats_simulation["rendement_total"]
#                 }

#                 if resultats_prediction["regression_metrics_temps_volume"]["r2"] is not None:
#                     resultats_strategies[f"DCA {frequence_courante.title()} (Temps+Volume)"] = {
#                         "timeline_historique": resultats_simulation["timeline"],
#                         "predictions": predictions_formatees,
#                         "regression_metrics": resultats_prediction["regression_metrics_temps_volume"],
#                         "residus": resultats_prediction["residus_temps_volume"],
#                         "cagr": resultats_simulation["cagr"],
#                         "rendement_total": resultats_simulation["rendement_total"]
#                     }

#                 # Modèle AR pour cette stratégie DCA
#                 resultats_ar_dca = self.predire_avec_processus_ar(chronologie_historique, lags=1, horizon=duree_prediction_ans*12)
#                 resultats_strategies[f"DCA {frequence_courante.title()} (Processus AR)"] = {
#                     "timeline_historique": resultats_simulation["timeline"],
#                     "predictions": resultats_ar_dca["predictions"],
#                     "ar_metrics": {
#                         "aic": resultats_ar_dca["aic"],
#                         "bic": resultats_ar_dca["bic"],
#                         "params": resultats_ar_dca["params"]
#                     },
#                     "residus": resultats_ar_dca["residus"],
#                     "cagr": resultats_simulation["cagr"],
#                     "rendement_total": resultats_simulation["rendement_total"]
#                 }

#             except Exception as erreur_dca:
#                 print(f"Erreur pour DCA {frequence_courante}: {erreur_dca}")
#                 continue

#         # Lump Sum
#         try:
#             resultats_lump_sum = self.simulateur.simulate_portfolio_growth(
#                 actifs=actifs,
#                 start=start,
#                 end=end,
#                 montant_initial=montant_total_disponible,
#                 apport_periodique=0.0,
#                 frequence="mensuelle",
#                 frais_gestion_pct=frais_gestion_pct
#             )

#             chronologie_lump_sum = pd.Series(resultats_lump_sum["timeline"])
#             chronologie_lump_sum.index = pd.to_datetime(chronologie_lump_sum.index)

#             predictions_lump_sum = self.predire_avec_regression(
#                 chronologie_lump_sum, 
#                 duree_prediction_ans, 
#                 volume_serie=volume_moyen
#             )

#             predictions_lump_formatees = {
#                 date_prediction.strftime("%Y-%m-%d"): float(valeur_predite)
#                 for date_prediction, valeur_predite in predictions_lump_sum["predictions"].items()
#             }

#             resultats_strategies["Lump Sum (Temps seul)"] = {
#                 "timeline_historique": resultats_lump_sum["timeline"],
#                 "predictions": predictions_lump_formatees,
#                 "regression_metrics": predictions_lump_sum["regression_metrics_temps"],
#                 "residus": predictions_lump_sum["residus_temps"],
#                 "cagr": resultats_lump_sum["cagr"],
#                 "rendement_total": resultats_lump_sum["rendement_total"]
#             }

#             if predictions_lump_sum["regression_metrics_temps_volume"]["r2"] is not None:
#                 resultats_strategies["Lump Sum (Temps+Volume)"] = {
#                     "timeline_historique": resultats_lump_sum["timeline"],
#                     "predictions": predictions_lump_formatees,
#                     "regression_metrics": predictions_lump_sum["regression_metrics_temps_volume"],
#                     "residus": predictions_lump_sum["residus_temps_volume"],
#                     "cagr": resultats_lump_sum["cagr"],
#                     "rendement_total": resultats_lump_sum["rendement_total"]
#                 }

#             resultats_ar_lump = self.predire_avec_processus_ar(chronologie_lump_sum, lags=1, horizon=duree_prediction_ans*12)
#             resultats_strategies["Lump Sum (Processus AR)"] = {
#                 "timeline_historique": resultats_lump_sum["timeline"],
#                 "predictions": resultats_ar_lump["predictions"],
#                 "ar_metrics": {
#                     "aic": resultats_ar_lump["aic"],
#                     "bic": resultats_ar_lump["bic"],
#                     "params": resultats_ar_lump["params"]
#                 },
#                 "residus": resultats_ar_lump["residus"],
#                 "cagr": resultats_lump_sum["cagr"],
#                 "rendement_total": resultats_lump_sum["rendement_total"]
#             }

#         except Exception as erreur_lump_sum:
#             print(f"Erreur pour Lump Sum: {erreur_lump_sum}")

#         return {"strategies": resultats_strategies}