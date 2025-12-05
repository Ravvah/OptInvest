from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from app.api.schemas.prediction import PredictionResponse
from app.api.schemas.simulation import SimulationResponse
import app.client.constants as cst


class DashboardManager:

    def __init__(self):
        pass

    def display_principal_metrics(self, donnees_simulation_portefeuille: SimulationResponse) -> None:
        """
        Affiche les métriques principales de performance.
        """

        # interface = UserForm()
        
        # TO DO: enlever car le cash investi est déjà dans les données retournées par l'API
        # cash_total_investi = interface.calculer_cash_investi(
        #     parametres["duree_simulation_annees"],
        #     parametres["frequence_investissement"],
        #     parametres["montant_initial"],
        #     parametres["apport_periodique"]
        # )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CAGR", f"{donnees_simulation_portefeuille.cagr*100:.2f} %")
        col2.metric("Rendement total", f"{donnees_simulation_portefeuille.rendement_total:.2f} €")
        col3.metric("Cash investi", f"{donnees_simulation_portefeuille.cash_investi:.2f} €")
        col4.metric("Volatilité ann.", f"{donnees_simulation_portefeuille.volatilite_annualisee*100:.2f} %")
        col5.metric("Sharpe", f"{donnees_simulation_portefeuille.ratio_sharpe:.2f}")

    def display_portfolio_vs_cash_plot(self, donnees_simulation_portefeuille: SimulationResponse) -> None:
        """
        Affiche le graphique principal du portefeuille vs cash investi.
        """
        valeur_portefeuille_temps_serie = pd.Series(donnees_simulation_portefeuille.valeur_portefeuille_temps, dtype=float)
        valeur_portefeuille_temps_serie.index = pd.to_datetime(valeur_portefeuille_temps_serie.index)
        
        valeur_cash_temps_serie = pd.Series(donnees_simulation_portefeuille.valeur_montant_temps, dtype=float)
        valeur_cash_temps_serie.index = pd.to_datetime(valeur_cash_temps_serie.index)
        
        st.subheader("Valeur du portefeuille vs cash investi")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valeur_portefeuille_temps_serie.index,
            y=valeur_portefeuille_temps_serie.values,
            mode='lines',
            name='Portefeuille',
            line=dict(color= cst.COULEURS_PALETTE[0], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=valeur_cash_temps_serie.index,
            y=valeur_cash_temps_serie.values,
            mode='lines',
            name='Cash investi',
            line=dict(color = cst.COULEURS_PALETTE[1], width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_portfolio_vs_world(self, donnees_simulation_portefeuille: SimulationResponse, valeur_portefeuille_temps_monde: dict[str, float]) -> None:
        """
        Affiche la comparaison portefeuille vs benchmark (déjà normalisés en euros).
        """
        st.subheader("Comparaison avec l'indice monde ACWI IMI")
        
        if valeur_portefeuille_temps_monde is None or len(valeur_portefeuille_temps_monde) == 0:
            st.warning("Impossible de charger les données de l'indice ACWI IMI.")
            return
        
        valeur_portefeuille_temps_serie = pd.Series(donnees_simulation_portefeuille.valeur_portefeuille_temps, dtype=float)
        valeur_portefeuille_temps_serie.index = pd.to_datetime(valeur_portefeuille_temps_serie.index)
        
        valeur_portefeuille_temps_monde_serie = pd.Series(valeur_portefeuille_temps_monde, dtype=float)
        valeur_portefeuille_temps_monde_serie.index = pd.to_datetime(valeur_portefeuille_temps_monde_serie.index)
        
        print(f"Portefeuille - Début: {valeur_portefeuille_temps_serie.iloc[0]:.2f}€, Fin: {valeur_portefeuille_temps_serie.iloc[-1]:.2f}€")
        print(f"Benchmark - Début: {valeur_portefeuille_temps_monde_serie.iloc[0]:.2f}€, Fin: {valeur_portefeuille_temps_monde_serie.iloc[-1]:.2f}€")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valeur_portefeuille_temps_serie.index,
            y=valeur_portefeuille_temps_serie.values,
            mode='lines',
            name='Votre portefeuille',
            line=dict(color=cst.COULEURS_PALETTE[0], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=valeur_portefeuille_temps_monde_serie.index,
            y=valeur_portefeuille_temps_monde_serie.values,
            mode='lines',
            name='Indice Monde ACWI IMI',
            line=dict(color=cst.COULEURS_PALETTE[2], width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Les deux courbes démarrent au même montant pour faciliter la comparaison.")

        # perf_portefeuille = ((valeur_portefeuille_temps_serie.iloc[-1] / valeur_portefeuille_temps_serie.iloc[0]) - 1) * 100
        # perf_benchmark = ((valeur_portefeuille_temps_monde_serie.iloc[-1] / valeur_portefeuille_temps_monde.iloc[0]) - 1) * 100
        
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Performance portefeuille", f"{perf_portefeuille:.2f}%")
        # col2.metric("Performance ACWI IMI", f"{perf_benchmark:.2f}%")
        # col3.metric("Différence", f"{perf_portefeuille - perf_benchmark:+.2f}%")
        

    def display_mensual_returns_distribution(self, donnees_simulation_portefeuille: SimulationResponse) -> None:
        """
        Affiche la distribution des rendements mensuels.
        """

        valeur_rendements_mensuels_temps_serie = pd.Series(donnees_simulation_portefeuille.valeur_rendements_mensuels_temps, dtype=float)
        valeur_rendements_mensuels_temps_serie.index = pd.to_datetime(valeur_rendements_mensuels_temps_serie.index)
        
        # rendements_mensuels = valeur_portefeuille_temps_serie.resample("ME").last().pct_change().dropna()
        
        st.subheader("Distribution des rendements mensuels")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=valeur_rendements_mensuels_temps_serie.index,
            y=np.array(valeur_rendements_mensuels_temps_serie.values) * 100,
            name='Rendements mensuels',
            marker_color=cst.COULEURS_PALETTE[3]
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_all_models_predictions(self, donnees_simulation_prediction_portefeuille: List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        """
        Affiche les prédictions et comparaisons DCA.
        """
        st.header("Comparaison des différentes stratégies DCA et Lump Sum sur les prédictions")

        onglet_regression, onglet_processus_ar, onglet_processus_arch = st.tabs(["Régression linéaire", "Processus AR", "Processus ARCH"])

        with onglet_regression:
            st.header("Valeur du portefeuille en fonction du montant investi")
            self.display_all_strategies_model_prediction_elements(donnees_simulation_prediction_portefeuille)

        #TO DO: Implémenter plus tard l'affichage avec AR
        with onglet_processus_ar:
            st.subheader("Modèle Processus Auto-Régressif AR(1)")
            st.info("Aucun modèle AR disponible.")

        #TO DO: Implémenter plus tard l'affichage avec ARCH
        with onglet_processus_arch:
            st.subheader("Modèle Processus ARCH")
            st.info("Aucun modèle ARCH disponible.")

    def display_all_strategies_model_prediction_elements(self, donnees_simulation_prediction_portefeuille: List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        """Affiche l'ensemble des éléments liés au modèle linéaire gaussien pour la stratégie selectionnée."""
        self.display_simulation_prediction_plot_strategies_comparison(donnees_simulation_prediction_portefeuille)
        self.display_prediction_quality_metrics(donnees_simulation_prediction_portefeuille)
        self.display_residual_analysis_metrics(donnees_simulation_prediction_portefeuille)

    def display_simulation_prediction_plot_strategies_comparison(self, donnees_simulation_prediction_strategies: List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        st.subheader("Comparaison des Stratégies sur les simulations et prédictions")
        fig = go.Figure()
        fig_2 = go.Figure()

        for index_strategie, (donnees_simulation, donnees_prediction) in enumerate(donnees_simulation_prediction_strategies):
            couleur_strategie = cst.COULEURS_PALETTE[index_strategie % len(cst.COULEURS_PALETTE)]
            if donnees_simulation.frequence != "lump sum":
                valeur_portefeuille_montant_simulation = pd.Series(donnees_simulation.valeur_portefeuille_montant, dtype=float)
                valeur_portefeuille_montant_simulation.index = valeur_portefeuille_montant_simulation.index.astype(float)
                valeur_portefeuille_montant_simulation = valeur_portefeuille_montant_simulation.sort_index()
                valeur_portefeuille_montant_prediction = pd.Series(donnees_prediction.predictions)
                valeur_portefeuille_montant_prediction.index = valeur_portefeuille_montant_prediction.index.astype(float)
                valeur_portefeuille_montant_prediction = valeur_portefeuille_montant_prediction.sort_index()

                fig.add_trace(go.Scatter(
                x=valeur_portefeuille_montant_simulation.index,
                y=valeur_portefeuille_montant_simulation.values,
                mode='lines',
                name=f'Strategie - {donnees_simulation.frequence} (Historique)',
                line=dict(color=couleur_strategie, width=2),
                showlegend=True
                ))
                fig.add_trace(go.Scatter(
                x=valeur_portefeuille_montant_prediction.index,
                y=valeur_portefeuille_montant_prediction.values,
                mode='lines',
                name=f'Strategie - {donnees_simulation.frequence} (Prédictions)',
                line=dict(color=couleur_strategie, width=2, dash='dash'),
                showlegend=True
                ))
            valeur_portefeuille_temps_simulation = pd.Series(donnees_simulation.valeur_portefeuille_temps, dtype=float)
            valeur_portefeuille_temps_simulation.index = pd.to_datetime(valeur_portefeuille_temps_simulation.index)
            valeur_portefeuille_temps_prediction = pd.Series(donnees_prediction.predictions_timeline, dtype=float)
            valeur_portefeuille_temps_prediction.index = pd.to_datetime(valeur_portefeuille_temps_prediction.index)


            fig_2.add_trace(go.Scatter(
                x=valeur_portefeuille_temps_simulation.index,
                y=valeur_portefeuille_temps_simulation.values,
                mode='lines',
                name=f'Strategie - {donnees_simulation.frequence} (Historique)',
                line=dict(color=couleur_strategie, width=2),
                showlegend=True
            ))
            fig_2.add_trace(go.Scatter(
                x=valeur_portefeuille_temps_prediction.index,
                y=valeur_portefeuille_temps_prediction.values,
                mode='lines',
                name=f'Strategie - {donnees_simulation.frequence} (Prédictions)',
                line=dict(color=couleur_strategie, width=2, dash='dash'),
                showlegend=True
            ))
        fig.update_layout(
            title=f"Évolution des Stratégies",
            xaxis_title="Montant investi cumulatif",
            yaxis_title="Valeur du Portefeuille (€)",
            height=500,
            template="plotly_white"
        )
        fig_2.update_layout(
            title=f"Évolution des Stratégies",
            xaxis_title="Temps",
            yaxis_title="Valeur du Portefeuille (€)",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig_2, use_container_width=True)


    def display_prediction_quality_metrics(self, donnees_simulation_prediction_strategies: List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        st.subheader(f"Métriques principales")

        colonnes_metriques = st.columns(len(donnees_simulation_prediction_strategies))
        for index_strategie, (donnees_simulation, donnees_prediction) in enumerate(donnees_simulation_prediction_strategies):
            with colonnes_metriques[index_strategie]:
                st.write(f"**{donnees_simulation.frequence}**")
                coefficient_determination = donnees_prediction.fit_quality_analysis.r2
                st.metric("R²", f"{coefficient_determination:.4f}")
                rmse_relative = donnees_prediction.fit_quality_analysis.rmse
                st.metric("RMSE", f"{rmse_relative:.2f}", delta=rmse_relative)

    def display_residual_analysis_metrics(self, donnees_simulation_prediction_strategies: List[Tuple[SimulationResponse, PredictionResponse]]) -> None:
        st.subheader(f"Diagnostic des résidus")
        tabs = st.tabs(list(cst.FREQUENCES.keys()))
        for tab, (_, donnee_prediction) in zip(tabs, donnees_simulation_prediction_strategies):
            with tab:
                self.display_residual_distribution_analysis(tab, donnee_prediction)

    def display_residual_distribution_analysis(self, nom_strategie: str, donnees_prediction: PredictionResponse) -> None:
        residus = donnees_prediction.residual_analysis.residuals
        col1, col2 = st.columns(2)
        ecart_type_residus = donnees_prediction.residual_analysis.ecart_type
        col1.metric(
            "Écart-type des résidus",
            f"{ecart_type_residus:.2f} €",
        )
        dw_stat = donnees_prediction.residual_analysis.autocorrelation_statistic
        
        col2.metric("Durbin-Watson", f"{dw_stat:.4f}")
        if dw_stat < 1.5:
            st.warning("Résidus positivement autocorrélés")
        elif dw_stat > 2.5:
            st.warning("Résidus négativement autocorrélés")
        else:
            st.success("Pas d'autocorrélation marquée des résidus")
        
        st.write("**Distribution des Résidus vs Distribution Normale**")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residus,
            nbinsx=15,
            name="Résidus",
            opacity=0.7,
            marker_color=cst.COULEURS_PALETTE[3],
            histnorm='probability density'
        ))
        x_norm = np.linspace(min(residus), max(residus), 100)
        y_norm = stats.norm.pdf(x_norm, 0, ecart_type_residus)
        fig_hist.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Distribution Normale',
            line=dict(color='red', width=2)
        ))
        fig_hist.update_layout(
            title=f"Évaluation visuelle de la normalité des résidus",
            xaxis_title="Résidus (€)",
            yaxis_title="Densité",
            height=450,
            template="plotly_white",
            showlegend=True
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.info("**Comment interpréter ce graphique** : Plus l'histogramme des résidus (en violet) se rapproche de la courbe normale théorique (en rouge), plus la distribution des résidus est normale, ce qui est souhaitable pour la validité du modèle.")

    def display_prediction_data(self, donnees_simulation: SimulationResponse, donnees_prediction: PredictionResponse) -> None:
        """
        Affiche un aperçu des données de simulation et de prédiction.
        """
        st.subheader("Aperçu des données")
        # Historique
        valeur_portefeuille_temps_simulation_serie = pd.Series(donnees_simulation.valeur_portefeuille_montant)
        df_valeur_portefeuille_temps_simulation = valeur_portefeuille_temps_simulation_serie.to_frame(name="Valeurs historiques en €")
        st.write("Données historiques (5 lignes) :")
        st.dataframe(df_valeur_portefeuille_temps_simulation.head())

        # Prédictions
        valeur_portefeuille_temps_prediction_serie = pd.Series(donnees_prediction.predictions_timeline)
        df_valeur_portefeuille_temps_prediction = valeur_portefeuille_temps_prediction_serie.to_frame("Valeurs futures en €")
        st.write("Données de prédictions (5 lignes) :")
        st.dataframe(df_valeur_portefeuille_temps_prediction.head())

    
    def display_markowitz_optimal_repartition(self, donnees_simulation: SimulationResponse):
        """
        Affiche une proposition de répartition optimale des actifs choisis au sens de Markowitz
        """
        st.header("Répartition optimale des actifs au sens de Markowitz")
        st.text("Les indicateurs sur la valeur du portefeuille de basent sur une répartition equitable des actifs séléctionnés")
        st.text("Voici une répartition optimale selon la théorie de Markowitz de la gestion des portefeuilles. Le modèle suppose de plus qu'on ne peut pas investir à découvert.")

        serie_poids = pd.Series(donnees_simulation.repartition_optimale_markowitz.repartition_actifs)
        df_repartition_optimale_markowitz = serie_poids.to_frame(name="Poids")
        print(donnees_simulation.repartition_optimale_markowitz.repartition_actifs)
        print(df_repartition_optimale_markowitz)
        st.write("Répartition optimale")
        st.dataframe(df_repartition_optimale_markowitz)

        col1, col2 = st.columns(2)
        
        col1.metric("Ratio de Sharpe optimal :", f"{donnees_simulation.repartition_optimale_markowitz.ratio_sharpe:.2f}")
        col2.metric("Volatilité annualisée :", f"{donnees_simulation.repartition_optimale_markowitz.volatilite_annualisee*100:.2f} %")





    # def _afficher_comparaison_strategies(self, donnees_strategies: Dict) -> None:
    #     """
    #     Affiche les courbes de comparaison des stratégies.
    #     """
    #     st.subheader("Comparaison des Stratégies d'Investissement")
        
    #     fig = go.Figure()
        
    #     for index_strategie, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
    #         couleur_strategie = cst.COULEURS_PALETTE[index_strategie % len(cst.COULEURS_PALETTE)]
            
    #         timeline_historique = pd.Series(donnees_strategie["timeline_historique"])
    #         timeline_historique.index = pd.to_datetime(timeline_historique.index)
            
    #         predictions_futures = pd.Series(donnees_strategie["predictions"])
    #         predictions_futures.index = pd.to_datetime(predictions_futures.index)
            
    #         fig.add_trace(go.Scatter(
    #             x=timeline_historique.index,
    #             y=timeline_historique.values,
    #             mode='lines',
    #             name=f'{nom_strategie} (Historique)',
    #             line=dict(color=couleur_strategie, width=2),
    #             showlegend=True
    #         ))
            
    #         fig.add_trace(go.Scatter(
    #             x=predictions_futures.index,
    #             y=predictions_futures.values,
    #             mode='lines',
    #             name=f'{nom_strategie} (Prédictions)',
    #             line=dict(color=couleur_strategie, width=2, dash='dash'),
    #             showlegend=True
    #         ))
        
    #     fig.update_layout(
    #         title="Évolution des Stratégies d'Investissement",
    #         xaxis_title="Date",
    #         yaxis_title="Valeur du Portefeuille (€)",
    #         height=500,
    #         template="plotly_white"
    #     )
        
    #     st.plotly_chart(fig, use_container_width=True)
        
    #     colonne_gauche, colonne_droite = st.columns(2)
    #     with colonne_gauche:
    #         st.info("**Ligne continue** : Données historiques")
    #     with colonne_droite:
    #         st.info("**Ligne pointillée** : Prédictions")

    # def _afficher_metriques_qualite(self, donnees_strategies: Dict) -> None:
    #     """
    #     Affiche les métriques principales de qualité des régressions.
    #     """
    #     st.subheader("Métriques principales")
        
    #     liste_strategies = list(donnees_strategies["strategies"].keys())
    #     colonnes_metriques = st.columns(len(liste_strategies))
        
    #     for index_colonne, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
    #         with colonnes_metriques[index_colonne]:
    #             st.write(f"**{nom_strategie}**")
                
    #             metriques_regression = donnees_strategie["regression_metrics"]
    #             timeline_historique = pd.Series(donnees_strategie["timeline_historique"], dtype=float)
    #             valeur_reference = float(timeline_historique.mean()) if len(timeline_historique) > 0 else float("nan")
                
    #             # Métriques principales
    #             st.metric("R²", f"{metriques_regression['r2']:.4f}")
    #             if valeur_reference > 0:
    #                 rmse_delta = f"{metriques_regression['rmse'] / valeur_reference * 100:.1f}% de la valeur moyenne"
    #                 mae_delta = f"{metriques_regression['mae'] / valeur_reference * 100:.1f}% de la valeur moyenne"
    #             else:
    #                 rmse_delta = None
    #                 mae_delta = None
    #             st.metric("RMSE", f"{metriques_regression['rmse']:.2f} €", delta=rmse_delta)
    #             st.metric("MAE", f"{metriques_regression['mae']:.2f} €", delta=mae_delta)

    #             # Évaluation globale de la qualité
    #             coefficient_determination = metriques_regression['r2']
    #             if coefficient_determination >= 0.8:
    #                 st.success("Excellente qualité")
    #             elif coefficient_determination >= 0.6:
    #                 st.warning("Qualité correcte")
    #             else:
    #                 st.error("Qualité faible")

    # def _afficher_diagnostic_residus(self, donnees_strategies: Dict) -> None:
    #     """
    #     Affiche le diagnostic complet des résidus (indicateurs + histogrammes).
    #     """
    #     st.subheader("Diagnostic des résidus")
        
    #     # Création des onglets pour chaque stratégie
    #     strategies_list = list(donnees_strategies["strategies"].keys())
    #     tabs = st.tabs(strategies_list)
        
    #     for tab, (nom_strategie, donnees_strategie) in zip(tabs, donnees_strategies["strategies"].items()):
    #         with tab:
    #             self._afficher_graphiques_residus_strategie(nom_strategie, donnees_strategie)

    # def _afficher_graphiques_residus_strategie(self, nom_strategie: str, donnees_strategie: Dict) -> None:
    #     """
    #     Affiche les graphiques de diagnostic pour une stratégie spécifique.
    #     """
    #     # Récupération des résidus
    #     metriques = donnees_strategie["regression_metrics"]
    #     timeline_hist = pd.Series(donnees_strategie["timeline_historique"])
        
    #     # Utiliser les vrais résidus si disponibles, sinon simuler
    #     if "residus" in donnees_strategie and donnees_strategie["residus"]:
    #         residus_reels = np.array(donnees_strategie["residus"])
    #     else:
    #         np.random.seed(42)
    #         n_points = len(timeline_hist)
    #         residus_reels = np.random.normal(0, metriques["std_residus"], n_points)

    #     col1, col2 = st.columns(2)
    #     ecart_type_residus = metriques["std_residus"]
    #     variation_portefeuille = float(timeline_hist.std()) if len(timeline_hist) > 1 else float("nan")
    #     if variation_portefeuille > 0:
    #         ratio_variation = ecart_type_residus / variation_portefeuille * 100
    #         delta_message = f"{ratio_variation:.1f}% de l'écart-type historique"
    #     else:
    #         delta_message = None

    #     col1.metric(
    #         "Écart-type des résidus",
    #         f"{ecart_type_residus:.2f} €",
    #         delta=delta_message
    #     )
    #     dw_stat = metriques.get("dw_stat")
    #     if dw_stat is not None:
    #         col2.metric("Durbin-Watson", f"{dw_stat:.4f}")
    #         if dw_stat < 1.5:
    #             st.warning("Résidus positivement autocorrélés")
    #         elif dw_stat > 2.5:
    #             st.warning("Résidus négativement autocorrélés")
    #         else:
    #             st.success("Pas d'autocorrélation marquée des résidus")
    #     else:
    #         col2.metric("Durbin-Watson", "N/A")
    #         st.info("Durbin-Watson non disponible pour cette stratégie")
        
    #     # Graphique: Distribution des résidus
    #     st.write("**Distribution des Résidus vs Distribution Normale**")
        
    #     fig_hist = go.Figure()
        
    #     fig_hist.add_trace(go.Histogram(
    #         x=residus_reels,
    #         nbinsx=15,
    #         name="Résidus",
    #         opacity=0.7,
    #         marker_color='#7209B7',
    #         histnorm='probability density'
    #     ))
        
    #     # Courbe normale théorique
    #     x_norm = np.linspace(residus_reels.min(), residus_reels.max(), 100)
    #     y_norm = stats.norm.pdf(x_norm, 0, metriques["std_residus"])
    #     fig_hist.add_trace(go.Scatter(
    #         x=x_norm,
    #         y=y_norm,
    #         mode='lines',
    #         name='Distribution Normale',
    #         line=dict(color='red', width=2)
    #     ))
        
    #     fig_hist.update_layout(
    #         title="Évaluation visuelle de la normalité des résidus",
    #         xaxis_title="Résidus (€)",
    #         yaxis_title="Densité",
    #         height=450,
    #         template="plotly_white",
    #         showlegend=True
    #     )
        
    #     st.plotly_chart(fig_hist, use_container_width=True)
        
    #     # Information d'aide sur la lecture du graphique
    #     st.info("**Comment interpréter ce graphique** : Plus l'histogramme des résidus (en violet) se rapproche de la courbe normale théorique (en rouge), plus la distribution des résidus est normale, ce qui est souhaitable pour la validité du modèle.")

    # def _afficher_metriques_ar_modele(self, donnees_strategies: Dict) -> None:
    #     st.subheader("Métriques principales - Processus AR")
    #     liste_strategies = list(donnees_strategies["strategies"].keys())
    #     if not liste_strategies:
    #         st.info("Aucune stratégie AR disponible.")
    #         return
    #     colonnes_metriques = st.columns(len(liste_strategies))
    #     for index_colonne, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
    #         ar_metrics = donnees_strategie.get("ar_metrics", {})
    #         with colonnes_metriques[index_colonne]:
    #             st.write(f"**{nom_strategie}**")
    #             st.metric("AIC", f"{ar_metrics.get('aic', float('nan')):.2f}")
    #             st.metric("BIC", f"{ar_metrics.get('bic', float('nan')):.2f}")
    #             params = ar_metrics.get("params", {})
    #             st.write("**Paramètres AR**")
    #             for k, v in params.items():
    #                 st.write(f"{k}: {v:.4f}")

    # def _afficher_diagnostic_residus_ar_modele(self, donnees_strategies: Dict) -> None:
    #     st.subheader("Diagnostic des résidus - Processus Auto-Régressif AR(1)")
    #     strategies_list = list(donnees_strategies["strategies"].keys())
    #     tabs = st.tabs(strategies_list)
    #     for tab, (nom_strategie, donnees_strategie) in zip(tabs, donnees_strategies["strategies"].items()):
    #         with tab:
    #             residus = np.array(donnees_strategie.get("residus", []))
    #             if residus.size == 0:
    #                 st.info("Pas de résidus disponibles pour cette stratégie.")
    #                 continue
    #             ecart_type_residus = float(np.std(residus, ddof=1)) if residus.size > 1 else float("nan")
    #             st.metric("Écart-type des résidus", f"{ecart_type_residus:.2f}")
    #             st.write("**Distribution des Résidus vs Distribution Normale**")
    #             fig_hist = go.Figure()
    #             fig_hist.add_trace(go.Histogram(
    #                 x=residus,
    #                 nbinsx=15,
    #                 name="Résidus",
    #                 opacity=0.7,
    #                 marker_color='#7209B7',
    #                 histnorm='probability density'
    #             ))
    #             x_norm = np.linspace(residus.min(), residus.max(), 100)
    #             y_norm = stats.norm.pdf(x_norm, 0, ecart_type_residus)
    #             fig_hist.add_trace(go.Scatter(
    #                 x=x_norm,
    #                 y=y_norm,
    #                 mode='lines',
    #                 name='Distribution Normale',
    #                 line=dict(color='red', width=2)
    #             ))
    #             fig_hist.update_layout(
    #                 title=f"Évaluation visuelle de la normalité des résidus - Processus AR",
    #                 xaxis_title="Résidus (€)",
    #                 yaxis_title="Densité",
    #                 height=450,
    #                 template="plotly_white",
    #                 showlegend=True
    #             )
    #             st.plotly_chart(fig_hist, use_container_width=True)
    #             st.info("**Comment interpréter ce graphique** : Plus l'histogramme des résidus (en violet) se rapproche de la courbe normale théorique (en rouge), plus la distribution des résidus est normale, ce qui est souhaitable pour la validité du modèle.")