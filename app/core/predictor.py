from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
from statsmodels.stats.stattools import durbin_watson
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
    
    def _get_r_squared_validation(self, y_true: List[float], y_pred: List[float]) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        ss_regression = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_total == 0.0:
            return 0.0
        
        r_squared_validation = 1 - (ss_regression / ss_total)
        return r_squared_validation

    
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
            r2_val = self._get_r_squared_validation(y_val, y_pred_val)
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
    def __init__(self, requete, lags):
        super().__init__(requete)
        self.model = None
        self.lags = lags        
        pass
    


#TO DO: implement an ARIMA Processus as the third model
class ARIMA1ModelPredictor(Predictor):
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None

#TO DO: implement an ARMA-GARCH Processus as the fourth model
class ARMAGARCH1ModelPredictor(Predictor):
    def __init__(self, requete):
        super().__init__(requete)
        self.model = None        
