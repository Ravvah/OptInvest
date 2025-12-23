from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from app.core.datasources.extract_yfinance import YahooExtractor
from app.api.schemas.simulation import MarkowitzOptimalResult, SimulationRequest, SimulationResponse
import app.api.schemas.constants as cst



class Simulator:
    """
    Simule un portefeuille passif.
    """

    def __init__(self, requete: SimulationRequest, extractor: YahooExtractor | None = YahooExtractor()) -> None:
        self.requete = requete
        self.extracteur = extractor or YahooExtractor()
        self.taux_sans_risque = self._get_empirical_risk_free_rate()

    
    def _load_individual_closes(self) -> pd.DataFrame:
        """
        Charge les prix de clôture de chaque actif pour l'analyse Markowitz.
        """
        df_prix = self.extracteur.concat_multiple_df(self.requete.actifs, 
                                                    self.requete.date_debut, 
                                                    self.requete.date_fin)
        
        cols_prix = [actif for actif in self.requete.actifs]
        return df_prix[cols_prix]

    def _load_average_closes(self) -> pd.Series[float]:
        """
        Charge les prix moyens pour une liste d'actifs.
        """

        logger.info(f"Chargement des prix moyens pour les actifs: {self.requete.actifs} de {self.requete.date_debut} à {self.requete.date_fin} pour la simulation ...")
        serie_prix_moyen = self.extracteur._fetch_average_close_and_volume(self.requete.actifs, self.requete.date_debut, self.requete.date_fin)["Prix_Moyen"]
        return serie_prix_moyen
    
    def _get_empirical_risk_free_rate(self) -> float:
        """
        Calcul le taux sans risque à considérer pour le ratio de Sharpe
        Il s'agit de la moyenne arithmétique sur le nombre d'années de simulation d'un taux souverain americain (US T-Bonds) 
        """
        df_prix_moyens_actif_sans_risque = self.extracteur.fetch_close_and_volume_actif(actif=cst.ACTIF_SANS_RISQUE, start=self.requete.date_debut, end=self.requete.date_fin)
        rendements_journaliers = df_prix_moyens_actif_sans_risque[cst.ACTIF_SANS_RISQUE].pct_change().dropna()
        taux_sans_risque_moyen_annuel = rendements_journaliers.mean() * cst.NOMBRE_JOURS_BOURSE_AN
        return taux_sans_risque_moyen_annuel


    def simulate_portfolio_growth(self) -> SimulationResponse:
        """
        Simule l'évolution d'un portefeuille avec apports périodiques en considérant que les actifs sont répartis équitablement, puis en proposant une optimisation de la répartition au sens Markowitz.
        """

        logger.info("Simulation du portefeuille ...")
        logger.info(f"Stratégie utilisée : {self.requete.strategie}")
        nombre_parts = 0.0
        cash_investi_total = 0.0
        valeur_portefeuille_temps: Dict[str, float] = {}
        valeurs_portefeuille_montants: Dict[str, float] = {}
        valeur_cash_cumule_temps: Dict[str, float] = {}
        is_lump_sum = self.requete.strategie == "lump sum"
        poids_equitables = np.array([1 / len(self.requete.actifs) for _ in range(len(self.requete.actifs))])

        if not is_lump_sum:
            nombre_mois = cst.FREQ2MONTH[self.requete.frequence]
            prochaine_echeance = pd.to_datetime(self.requete.date_debut) + pd.DateOffset(months=nombre_mois)
        montant_initial_investi = False

        prix_moyens = self._load_average_closes()
        assert isinstance(prix_moyens.index, pd.DatetimeIndex), "attendu un DatetimeIndex"

        dernier_mois = None
        for date_courante, prix_courant in prix_moyens.items():
            mois_courant = date_courante.month
            date_courante_convertie = date_courante.date()

            if not montant_initial_investi:
                cash_investi_total += self.requete.montant_initial
                nombre_parts += self.requete.montant_initial / prix_courant
                valeurs_portefeuille_montants[str(self.requete.montant_initial)] = nombre_parts * prix_courant
                montant_initial_investi = True
            
            elif not is_lump_sum and date_courante >= prochaine_echeance :
                nombre_parts += self.requete.apport_periodique / prix_courant
                cash_investi_total += self.requete.apport_periodique
                prochaine_echeance += pd.DateOffset(months=nombre_mois)

            # Frais de gestion annuels
            if dernier_mois == 12 and mois_courant == 1:
                nombre_parts *= 1 - self.requete.frais_gestion / 100

            valeur_portefeuille_temps[str(date_courante_convertie)] = nombre_parts * prix_courant
            valeurs_portefeuille_montants[str(cash_investi_total)] = nombre_parts * prix_courant
            valeur_cash_cumule_temps[str(date_courante_convertie)] = cash_investi_total
            dernier_mois = mois_courant


        if not valeur_portefeuille_temps or not valeurs_portefeuille_montants:
            raise ValueError("Aucune donnée de prix disponible pour les actifs et la période spécifiés.")
        
        valeur_portefeuille_serie = pd.Series(valeur_portefeuille_temps, dtype=float)
        valeur_portefeuille_serie.index = pd.to_datetime(valeur_portefeuille_serie.index)
        valeur_portefeuille_mensuel = valeur_portefeuille_serie.resample("ME").last().dropna()
        rendements_mensuels = valeur_portefeuille_mensuel.pct_change().dropna()
        rendements_mensuels.index = rendements_mensuels.index.map(lambda t: t.strftime("%Y-%m-%d"))
        valeur_rendements_mensuels_temps = rendements_mensuels.to_dict()

        date_fin_effective_str = list(valeur_portefeuille_temps)[-1]
        valeur_finale_portefeuille = valeur_portefeuille_temps[date_fin_effective_str]

        # duree_simulation_annees = (pd.to_datetime(date_fin_effective_str).date() - self.requete.date_debut).days / 365
        taux_croissance_annuel = self._get_cagr(valeur_initiale=cash_investi_total, valeur_finale=valeur_finale_portefeuille)
        
        rendement_net_total = valeur_finale_portefeuille - cash_investi_total

        df_valeurs_actifs = self._load_individual_closes()
        rendements_moyens_portefeuille = self._get_rendements_moyens_actifs(df_valeurs_actifs=df_valeurs_actifs)
        matrice_covariance_actifs = self._get_covariance_rendements_actifs(df_valeurs_actifs=df_valeurs_actifs)
        rendements_moyens_portefeuille_equitable = np.dot(poids_equitables, rendements_moyens_portefeuille)
        volatilite_annualisee = self._get_portfolio_global_volatility(poids=poids_equitables, matrice_covariance_actifs=matrice_covariance_actifs)
        ratio_sharpe = self._get_sharpe_ratio(rendement_moyen=rendements_moyens_portefeuille_equitable, volatilite_annualisee=volatilite_annualisee)
        repartition_optimale_markowitz = self._get_markowitz_optimal_weight_repartition(rendements_moyens_actifs=rendements_moyens_portefeuille, matrice_covariance_actifs=matrice_covariance_actifs)


        return SimulationResponse(
            frequence=self.requete.frequence,
            cash_investi=cash_investi_total,
            cagr=round(taux_croissance_annuel, 4),
            rendement_total=round(rendement_net_total, 2),
            volatilite_annualisee=round(volatilite_annualisee, 2),
            ratio_sharpe=round(ratio_sharpe, 2),
            repartition_optimale_markowitz=repartition_optimale_markowitz,
            valeur_portefeuille_temps=valeur_portefeuille_temps,
            valeur_portefeuille_montant=valeurs_portefeuille_montants,
            valeur_montant_temps=valeur_cash_cumule_temps,
            valeur_rendements_mensuels_temps=valeur_rendements_mensuels_temps
        )
    
    def _get_rendements_moyens_actifs(self, df_valeurs_actifs: pd.DataFrame) -> np.ndarray:
        rendements_journaliers = df_valeurs_actifs.pct_change().dropna()
        rendements_moyens_annualisee = rendements_journaliers.mean() * cst.NOMBRE_JOURS_BOURSE_AN
        return rendements_moyens_annualisee.values

    def _get_covariance_rendements_actifs(self, df_valeurs_actifs: pd.DataFrame) -> np.ndarray:
        rendements_journaliers = df_valeurs_actifs.pct_change().dropna()
        covariance_rendements_annualisee = rendements_journaliers.cov() * cst.NOMBRE_JOURS_BOURSE_AN
        return covariance_rendements_annualisee.values

    def _get_cagr(
        self,
        valeur_initiale: float,
        valeur_finale: float,
    ) -> float:
        """
        Calcule le CAGR.
        """
        if self.requete.duree_ans <= 0:
            return 0.0
        return (valeur_finale / valeur_initiale) ** (1 / self.requete.duree_ans) - 1

    def _get_actif_volatility(
        self,
        liste_valeurs: List[float],
        nb_periodes_an: int
    ) -> float:
        """
        Calcule la volatilité annualisée pour un actif.
        """
        rendements_periodiques_pourcentages = pd.Series(liste_valeurs).pct_change().dropna()
        volatilite_annualisee_actif = rendements_periodiques_pourcentages.std() * np.sqrt(nb_periodes_an)
        return volatilite_annualisee_actif
    
    def _get_portfolio_global_volatility(self, poids: np.ndarray, matrice_covariance_actifs: np.ndarray) -> float:
        """
        Calcule la volatilité annualisée du portefeuille global.
        """
        variance_portefeuille = np.dot(np.dot(poids.T, matrice_covariance_actifs), poids)
        volatilite_portefeuille = np.sqrt(variance_portefeuille)
        return volatilite_portefeuille
    
    def _get_sharpe_ratio(
        self,
        rendement_moyen: float,
        volatilite_annualisee: float,
    ) -> float:
        """
        Calcule le ratio de Sharpe.
        """
        if volatilite_annualisee <= 0:
            return 0.0
        
        return (rendement_moyen - self.taux_sans_risque) / volatilite_annualisee
    

    def _get_markowitz_optimal_weight_repartition(self, rendements_moyens_actifs: np.ndarray, matrice_covariance_actifs: np.ndarray) -> MarkowitzOptimalResult:
        """
        Calcule les poids optimaux des actifs dans le portefeuille par optimisation de la frontière efficiente (Théorie de Markowitz).
        Il s'agit d'un problème d'optimisation quadratique de l'utilité Moyenne-Variance, résolu par la méthode des points intérieurs.
        L'algorithme utilise une stratégie de balayage (en faisant varier un paramètre d'aversion au risque gamma) pour générer itérativement les portefeuilles qui se trouvent sur la frontière efficiente.
        Pour chaque valeur du paramètre, il itère pour ajuster les poids des actifs dans l'espace du Simplex (région faisable) jusqu'à convergence sur chaque point optimal de la frontière.
        La solution finale sera le point parmi les points optimaux sur la frontière efficiente qui maximise le ratio de Sharpe du portefeuille (le portefeuille tangent).
        Hypothèses : 
            - Les rendements des actifs sont normalement distribués.
            - Les rendements moyens des actifs et les covariances de rendements des actifs ne changent pas dans le temps.
            - La matrice de covariance des rendements des actifs est inversible.
            - Les investisseurs sont rationnels et cherchent à maximiser leur utilité.
        Contraintes :
            - La somme des poids doit être égale à 1.
            - Les poids doivent être non négatifs (pas de vente à découvert).
        """
        nombre_actifs = len(rendements_moyens_actifs)
        poids_initiaux = np.array([1 / nombre_actifs for _ in range(nombre_actifs)])
        intervalle_aversion_risque_gamma = np.logspace(start=-2, stop=2, num=50)
        contrainte_positivite_poids = tuple((0, 1) for _ in range(nombre_actifs))
        contrainte_budget_poids = {
            "type": "eq",
            "fun": lambda poids: np.sum(poids) - 1 
        }
        ensemble_poids_optimaux_frontiere = []

        for gamma in intervalle_aversion_risque_gamma:
            resulat_optimal = minimize(fun= self._objective_function,
                                      args=(rendements_moyens_actifs, matrice_covariance_actifs, gamma),
                                      x0=poids_initiaux,
                                      bounds=contrainte_positivite_poids,
                                      constraints=contrainte_budget_poids)
            if resulat_optimal.success:
                print("POIDS OPTIMAL ------------------- ")

                poids_optimaux = resulat_optimal.x
                print(poids_optimaux)

                rendement_moyen = np.dot(poids_optimaux, rendements_moyens_actifs)
                volatilite = self._get_portfolio_global_volatility(poids=poids_optimaux, matrice_covariance_actifs=matrice_covariance_actifs)

                ratio_sharpe = self._get_sharpe_ratio(rendement_moyen=rendement_moyen, volatilite_annualisee=volatilite)
                ensemble_poids_optimaux_frontiere.append({
                    "poids": poids_optimaux,
                    "ratio_sharpe": ratio_sharpe,
                    "rendement_moyen": rendement_moyen,
                    "volatilite": volatilite
                })
        
        if not ensemble_poids_optimaux_frontiere:
            logger.warning("Frontière optimale vide. Aucun poids trouvé. Verifiez l'algorithme")
            return {}
        
        print("FRONTIERE ---------------------- ")
        print(ensemble_poids_optimaux_frontiere)
        resultats_optimaux_tangents = max(ensemble_poids_optimaux_frontiere, key=lambda ensemble: ensemble["ratio_sharpe"])
        print("MAXIMUM ------------------------ ")
        print(resultats_optimaux_tangents)
        assert isinstance(resultats_optimaux_tangents, dict)

        ratio_sharpe_optimal = resultats_optimaux_tangents["ratio_sharpe"]
        rendement_moyen_optimal = resultats_optimaux_tangents["rendement_moyen"]
        volatilite_optimale = resultats_optimaux_tangents["volatilite"]
        poids_optimaux_tangents_list = resultats_optimaux_tangents["poids"].tolist()
        if len(self.requete.actifs) != len(poids_optimaux_tangents_list):
            raise ValueError("Le nombre d'actif et le nombre de poids sont différent !")

        actifs_poids_optimaux_dict = dict(zip(self.requete.actifs, poids_optimaux_tangents_list))

        return MarkowitzOptimalResult(
            repartition_actifs= actifs_poids_optimaux_dict,
            ratio_sharpe=round(ratio_sharpe_optimal, 2),
            rendement_moyen= round(rendement_moyen_optimal, 2),
            volatilite_annualisee= round(volatilite_optimale, 2) 
        )


    def _objective_function(self, poids: np.ndarray, rendements_moyens_actifs: np.ndarray, matrice_covariance_actifs: np.ndarray, gamma_aversion_risque: float) -> float:
        """
        Fonction objective pour l'optimisation quadratique de l'utilité Moyenne-Variance à minimiser.
        """

        rendement_portefeuille = np.dot(poids, rendements_moyens_actifs)
        risque_portefeuille = np.dot(np.dot(poids.T, matrice_covariance_actifs), poids)
        utilite = rendement_portefeuille - ((gamma_aversion_risque / 2) * risque_portefeuille)
        return (- utilite)

