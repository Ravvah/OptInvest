from __future__ import annotations

from typing import List, Tuple
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
                self.display_residual_distribution_analysis(donnee_prediction)

    def display_residual_distribution_analysis(self, donnees_prediction: PredictionResponse) -> None:
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
        valeur_portefeuille_montant_simulation_serie = pd.Series(donnees_simulation.valeur_portefeuille_montant)
        df_valeur_portefeuille_temps_simulation = valeur_portefeuille_montant_simulation_serie.to_frame(name="Valeurs historiques en €")
        st.write("Données historiques (5 lignes) :")
        st.dataframe(df_valeur_portefeuille_temps_simulation.head())

        # Prédictions
        valeur_portefeuille_montant_prediction_serie = pd.Series(donnees_prediction.predictions)
        df_valeur_portefeuille_temps_prediction = valeur_portefeuille_montant_prediction_serie.to_frame("Valeurs futures en €")
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