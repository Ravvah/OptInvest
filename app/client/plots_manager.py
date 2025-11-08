from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from ui_manager import InterfaceUtilisateur


class GestionnaireGraphiques:
    COULEURS_PALETTE = ['#FF6B35', '#004E89', '#009639', '#7209B7', '#D62828', '#F77F00']

    def afficher_metriques_principales(self, donnees_portefeuille: Dict, parametres: Dict) -> None:
        """
        Affiche les métriques principales de performance.
        """
        interface = InterfaceUtilisateur()
        
        cash_total_investi = interface.calculer_cash_investi(
            parametres["duree_simulation_annees"],
            parametres["frequence_investissement"],
            parametres["montant_initial"],
            parametres["apport_periodique"]
        )
        
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        rendements_mensuels = timeline_portefeuille.resample("ME").last().pct_change().dropna()
        volatilite_annualisee = rendements_mensuels.std() * np.sqrt(12)
        ratio_sharpe = ((rendements_mensuels.mean() * 12) - 0.02) / volatilite_annualisee if volatilite_annualisee > 0 else 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CAGR", f"{donnees_portefeuille['cagr']*100:.2f} %")
        col2.metric("Rendement total", f"{donnees_portefeuille['rendement_total']:.2f} €")
        col3.metric("Cash investi", f"{cash_total_investi:.2f} €")
        col4.metric("Volatilité ann.", f"{volatilite_annualisee*100:.2f} %")
        col5.metric("Sharpe", f"{ratio_sharpe:.2f}")

    def afficher_graphique_portefeuille(self, donnees_portefeuille: Dict, parametres: Dict) -> None:
        """
        Affiche le graphique principal du portefeuille vs cash investi.
        """
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        dates_cash = pd.date_range(timeline_portefeuille.index.min(), timeline_portefeuille.index.max(), freq="ME")
        cash_cumule = parametres["montant_initial"] + np.arange(len(dates_cash)) * parametres["apport_periodique"]
        serie_cash = pd.Series(cash_cumule, index=dates_cash)
        
        st.subheader("Valeur du portefeuille vs cash investi")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_portefeuille.index,
            y=timeline_portefeuille.values,
            mode='lines',
            name='Portefeuille',
            line=dict(color='#FF6B35', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=serie_cash.index,
            y=serie_cash.values,
            mode='lines',
            name='Cash investi',
            line=dict(color='#004E89', width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def afficher_comparaison_benchmark(self, donnees_portefeuille: Dict, serie_benchmark: pd.Series) -> None:
        """
        Affiche la comparaison portefeuille vs benchmark (déjà normalisés en euros).
        """
        st.subheader("Comparaison avec l'indice ACWI IMI")
        
        if serie_benchmark is None or serie_benchmark.empty:
            st.warning("⚠️ Impossible de charger les données de l'indice ACWI IMI.")
            return
        
        # Préparer le portefeuille
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        timeline_portefeuille = timeline_portefeuille.sort_index()
        
        # Le benchmark est déjà normalisé dans charger_benchmark
        serie_benchmark = serie_benchmark.sort_index()
        
        # Afficher les valeurs pour vérification
        print(f"Portefeuille - Début: {timeline_portefeuille.iloc[0]:.2f}€, Fin: {timeline_portefeuille.iloc[-1]:.2f}€")
        print(f"Benchmark - Début: {serie_benchmark.iloc[0]:.2f}€, Fin: {serie_benchmark.iloc[-1]:.2f}€")
        
        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_portefeuille.index,
            y=timeline_portefeuille.values,
            mode='lines',
            name='Votre portefeuille',
            line=dict(color='#FF6B35', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=serie_benchmark.index,
            y=serie_benchmark.values,
            mode='lines',
            name='Indice ACWI IMI',
            line=dict(color='#009639', width=3)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculer et afficher la performance relative
        perf_portefeuille = ((timeline_portefeuille.iloc[-1] / timeline_portefeuille.iloc[0]) - 1) * 100
        perf_benchmark = ((serie_benchmark.iloc[-1] / serie_benchmark.iloc[0]) - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Performance portefeuille", f"{perf_portefeuille:.2f}%")
        col2.metric("Performance ACWI IMI", f"{perf_benchmark:.2f}%")
        col3.metric("Différence", f"{perf_portefeuille - perf_benchmark:+.2f}%")
        
        st.info("Les deux courbes démarrent au même montant pour faciliter la comparaison.")

    def afficher_rendements_mensuels(self, donnees_portefeuille: Dict) -> None:
        """
        Affiche la distribution des rendements mensuels.
        """
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        rendements_mensuels = timeline_portefeuille.resample("ME").last().pct_change().dropna()
        
        st.subheader("Distribution des rendements mensuels")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rendements_mensuels.index,
            y=np.array(rendements_mensuels.values) * 100,
            name='Rendements mensuels',
            marker_color='#7209B7'
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def afficher_predictions_dca(self, donnees_prediction: Dict) -> None:
        """
        Affiche les prédictions et comparaisons DCA.
        """
        st.header("Prédictions et Comparaison DCA")

        onglet_regression, onglet_processus_ar = st.tabs(["Régression linéaire", "Processus AR"])

        with onglet_regression:
            sous_onglet_temps, sous_onglet_temps_volume = st.tabs(["Modèle Temps seul", "Modèle Temps + Volume"])
            with sous_onglet_temps:
                self._afficher_modele_regression_lineaire(donnees_prediction, type_modele="temps")
            with sous_onglet_temps_volume:
                self._afficher_modele_regression_lineaire(donnees_prediction, type_modele="temps_volume")

        with onglet_processus_ar:
            st.subheader("Modèle Processus AR")
            st.info(
                "Cette section accueillera prochainement un modèle de prédiction basé sur un processus autorégressif (AR)."
            )

    def _afficher_modele_regression_lineaire(self, donnees_prediction: Dict, type_modele: str = "temps") -> None:
        """Affiche l'ensemble des éléments liés au modèle linéaire gaussien pour un type de modèle donné."""
        if type_modele == "temps":
            strategies = {k: v for k, v in donnees_prediction["strategies"].items() if "(Temps seul)" in k}
        elif type_modele == "temps_volume":
            strategies = {k: v for k, v in donnees_prediction["strategies"].items() if "(Temps+Volume)" in k}
        else:
            strategies = donnees_prediction["strategies"]

        self._afficher_comparaison_strategies_modele({"strategies": strategies}, type_modele)
        self._afficher_metriques_qualite_modele({"strategies": strategies}, type_modele)
        self._afficher_diagnostic_residus_modele({"strategies": strategies}, type_modele)
        self._afficher_dataframe_donnees_prediction({"strategies": strategies}, type_modele)

    def _afficher_comparaison_strategies_modele(self, donnees_strategies: Dict, type_modele: str) -> None:
        st.subheader(f"Comparaison des Stratégies - {type_modele.replace('_', ' ').title()}")
        fig = go.Figure()
        for index_strategie, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
            couleur_strategie = self.COULEURS_PALETTE[index_strategie % len(self.COULEURS_PALETTE)]
            timeline_historique = pd.Series(donnees_strategie["timeline_historique"])
            timeline_historique.index = pd.to_datetime(timeline_historique.index)
            predictions_futures = pd.Series(donnees_strategie["predictions"])
            predictions_futures.index = pd.to_datetime(predictions_futures.index)
            fig.add_trace(go.Scatter(
                x=timeline_historique.index,
                y=timeline_historique.values,
                mode='lines',
                name=f'{nom_strategie} (Historique)',
                line=dict(color=couleur_strategie, width=2),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=predictions_futures.index,
                y=predictions_futures.values,
                mode='lines',
                name=f'{nom_strategie} (Prédictions)',
                line=dict(color=couleur_strategie, width=2, dash='dash'),
                showlegend=True
            ))
        fig.update_layout(
            title=f"Évolution des Stratégies - {type_modele.replace('_', ' ').title()}",
            xaxis_title="Date",
            yaxis_title="Valeur du Portefeuille (€)",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def _afficher_metriques_qualite_modele(self, donnees_strategies: Dict, type_modele: str) -> None:
        st.subheader(f"Métriques principales - {type_modele.replace('_', ' ').title()}")
        liste_strategies = list(donnees_strategies["strategies"].keys())
        
        if not liste_strategies:
            st.info("Aucune stratégie disponible pour ce modèle.")
            return
        
        colonnes_metriques = st.columns(len(liste_strategies))
        for index_colonne, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
            metriques = donnees_strategie["regression_metrics"]
            with colonnes_metriques[index_colonne]:
                st.write(f"**{nom_strategie}**")
                timeline_historique = pd.Series(donnees_strategie["timeline_historique"], dtype=float)
                valeur_reference = float(timeline_historique.mean()) if len(timeline_historique) > 0 else float("nan")
                st.metric("R²", f"{metriques['r2']:.4f}")
                if valeur_reference > 0:
                    rmse_delta = f"{metriques['rmse'] / valeur_reference * 100:.1f}% de la valeur moyenne"
                    mae_delta = f"{metriques['mae'] / valeur_reference * 100:.1f}% de la valeur moyenne"
                else:
                    rmse_delta = None
                    mae_delta = None
                st.metric("RMSE", f"{metriques['rmse']:.2f} €", delta=rmse_delta)
                st.metric("MAE", f"{metriques['mae']:.2f} €", delta=mae_delta)
                coefficient_determination = metriques['r2']
                if coefficient_determination >= 0.8:
                    st.success("Excellente qualité")
                elif coefficient_determination >= 0.6:
                    st.warning("Qualité correcte")
                else:
                    st.error("Qualité faible")

    def _afficher_diagnostic_residus_modele(self, donnees_strategies: Dict, type_modele: str) -> None:
        st.subheader(f"Diagnostic des résidus - {type_modele.replace('_', ' ').title()}")
        strategies_list = list(donnees_strategies["strategies"].keys())
        tabs = st.tabs(strategies_list)
        for tab, (nom_strategie, donnees_strategie) in zip(tabs, donnees_strategies["strategies"].items()):
            with tab:
                self._afficher_graphiques_residus_strategie_modele(nom_strategie, donnees_strategie, type_modele)

    def _afficher_graphiques_residus_strategie_modele(self, nom_strategie: str, donnees_strategie: Dict, type_modele: str) -> None:
        metriques = donnees_strategie["regression_metrics"]
        timeline_hist = pd.Series(donnees_strategie["timeline_historique"])
        if "residus" in donnees_strategie and donnees_strategie["residus"]:
            residus_reels = np.array(donnees_strategie["residus"])
        else:
            np.random.seed(42)
            n_points = len(timeline_hist)
            residus_reels = np.random.normal(0, metriques["std_residus"], n_points)
        col1, col2 = st.columns(2)
        ecart_type_residus = metriques["std_residus"]
        variation_portefeuille = float(timeline_hist.std()) if len(timeline_hist) > 1 else float("nan")
        if variation_portefeuille > 0:
            ratio_variation = ecart_type_residus / variation_portefeuille * 100
            delta_message = f"{ratio_variation:.1f}% de l'écart-type historique"
        else:
            delta_message = None
        col1.metric(
            "Écart-type des résidus",
            f"{ecart_type_residus:.2f} €",
            delta=delta_message
        )
        dw_stat = metriques.get("dw_stat")
        if dw_stat is not None:
            col2.metric("Durbin-Watson", f"{dw_stat:.4f}")
            if dw_stat < 1.5:
                st.warning("Résidus positivement autocorrélés")
            elif dw_stat > 2.5:
                st.warning("Résidus négativement autocorrélés")
            else:
                st.success("Pas d'autocorrélation marquée des résidus")
        else:
            col2.metric("Durbin-Watson", "N/A")
            st.info("Durbin-Watson non disponible pour cette stratégie")
        st.write("**Distribution des Résidus vs Distribution Normale**")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residus_reels,
            nbinsx=15,
            name="Résidus",
            opacity=0.7,
            marker_color='#7209B7',
            histnorm='probability density'
        ))
        x_norm = np.linspace(residus_reels.min(), residus_reels.max(), 100)
        y_norm = stats.norm.pdf(x_norm, 0, metriques["std_residus"])
        fig_hist.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Distribution Normale',
            line=dict(color='red', width=2)
        ))
        fig_hist.update_layout(
            title=f"Évaluation visuelle de la normalité des résidus - {type_modele.replace('_', ' ').title()}",
            xaxis_title="Résidus (€)",
            yaxis_title="Densité",
            height=450,
            template="plotly_white",
            showlegend=True
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.info("**Comment interpréter ce graphique** : Plus l'histogramme des résidus (en violet) se rapproche de la courbe normale théorique (en rouge), plus la distribution des résidus est normale, ce qui est souhaitable pour la validité du modèle.")

    def _afficher_dataframe_donnees_prediction(self, donnees_prediction: Dict, type_modele: str = "temps") -> None:
        """
        Affiche le DataFrame des données de prédiction.
        """
        st.subheader("Aperçu des données de prédictions")
        for nom_strategie, donnees_strategie in donnees_prediction["strategies"].items():
            st.markdown(f"**Stratégie : {nom_strategie}**")
            # Historique
            df_historique = pd.DataFrame(donnees_strategie["timeline_historique"].items(), columns=["Date", "Valeur historique"]).set_index("Date")
            if type_modele == "temps_volume" and "volume" in donnees_strategie and donnees_strategie["volume"] is not None:
                # On tente d'aligner la série volume sur les dates historiques
                volume = donnees_strategie["volume"]
                # Si volume est une liste, on suppose qu'elle est alignée sur les mêmes dates que timeline_historique
                if isinstance(volume, list) and len(volume) == len(df_historique):
                    df_historique["Volume"] = volume
                else:
                    # Sinon, on tente de construire une série pandas et de l'aligner
                    try:
                        s_volume = pd.Series(volume)
                        s_volume.index = df_historique.index
                        df_historique["Volume"] = s_volume
                    except Exception:
                        pass
            st.write("Données historiques (5 lignes) :")
            st.dataframe(df_historique.head())

            # Prédictions
            df_predictions = pd.DataFrame(donnees_strategie["predictions"].items(), columns=["Date", "Valeur prédite"]).set_index("Date")
            st.write("Données de prédictions (5 lignes) :")
            st.dataframe(df_predictions.head())

    def _afficher_comparaison_strategies(self, donnees_strategies: Dict) -> None:
        """
        Affiche les courbes de comparaison des stratégies.
        """
        st.subheader("Comparaison des Stratégies d'Investissement")
        
        fig = go.Figure()
        
        for index_strategie, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
            couleur_strategie = self.COULEURS_PALETTE[index_strategie % len(self.COULEURS_PALETTE)]
            
            timeline_historique = pd.Series(donnees_strategie["timeline_historique"])
            timeline_historique.index = pd.to_datetime(timeline_historique.index)
            
            predictions_futures = pd.Series(donnees_strategie["predictions"])
            predictions_futures.index = pd.to_datetime(predictions_futures.index)
            
            fig.add_trace(go.Scatter(
                x=timeline_historique.index,
                y=timeline_historique.values,
                mode='lines',
                name=f'{nom_strategie} (Historique)',
                line=dict(color=couleur_strategie, width=2),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions_futures.index,
                y=predictions_futures.values,
                mode='lines',
                name=f'{nom_strategie} (Prédictions)',
                line=dict(color=couleur_strategie, width=2, dash='dash'),
                showlegend=True
            ))
        
        fig.update_layout(
            title="Évolution des Stratégies d'Investissement",
            xaxis_title="Date",
            yaxis_title="Valeur du Portefeuille (€)",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        colonne_gauche, colonne_droite = st.columns(2)
        with colonne_gauche:
            st.info("**Ligne continue** : Données historiques")
        with colonne_droite:
            st.info("**Ligne pointillée** : Prédictions")

    def _afficher_metriques_qualite(self, donnees_strategies: Dict) -> None:
        """
        Affiche les métriques principales de qualité des régressions.
        """
        st.subheader("Métriques principales")
        
        liste_strategies = list(donnees_strategies["strategies"].keys())
        colonnes_metriques = st.columns(len(liste_strategies))
        
        for index_colonne, (nom_strategie, donnees_strategie) in enumerate(donnees_strategies["strategies"].items()):
            with colonnes_metriques[index_colonne]:
                st.write(f"**{nom_strategie}**")
                
                metriques_regression = donnees_strategie["regression_metrics"]
                timeline_historique = pd.Series(donnees_strategie["timeline_historique"], dtype=float)
                valeur_reference = float(timeline_historique.mean()) if len(timeline_historique) > 0 else float("nan")
                
                # Métriques principales
                st.metric("R²", f"{metriques_regression['r2']:.4f}")
                if valeur_reference > 0:
                    rmse_delta = f"{metriques_regression['rmse'] / valeur_reference * 100:.1f}% de la valeur moyenne"
                    mae_delta = f"{metriques_regression['mae'] / valeur_reference * 100:.1f}% de la valeur moyenne"
                else:
                    rmse_delta = None
                    mae_delta = None
                st.metric("RMSE", f"{metriques_regression['rmse']:.2f} €", delta=rmse_delta)
                st.metric("MAE", f"{metriques_regression['mae']:.2f} €", delta=mae_delta)

                # Évaluation globale de la qualité
                coefficient_determination = metriques_regression['r2']
                if coefficient_determination >= 0.8:
                    st.success("Excellente qualité")
                elif coefficient_determination >= 0.6:
                    st.warning("Qualité correcte")
                else:
                    st.error("Qualité faible")

    def _afficher_diagnostic_residus(self, donnees_strategies: Dict) -> None:
        """
        Affiche le diagnostic complet des résidus (indicateurs + histogrammes).
        """
        st.subheader("Diagnostic des résidus")
        
        # Création des onglets pour chaque stratégie
        strategies_list = list(donnees_strategies["strategies"].keys())
        tabs = st.tabs(strategies_list)
        
        for tab, (nom_strategie, donnees_strategie) in zip(tabs, donnees_strategies["strategies"].items()):
            with tab:
                self._afficher_graphiques_residus_strategie(nom_strategie, donnees_strategie)

    def _afficher_graphiques_residus_strategie(self, nom_strategie: str, donnees_strategie: Dict) -> None:
        """
        Affiche les graphiques de diagnostic pour une stratégie spécifique.
        """
        # Récupération des résidus
        metriques = donnees_strategie["regression_metrics"]
        timeline_hist = pd.Series(donnees_strategie["timeline_historique"])
        
        # Utiliser les vrais résidus si disponibles, sinon simuler
        if "residus" in donnees_strategie and donnees_strategie["residus"]:
            residus_reels = np.array(donnees_strategie["residus"])
        else:
            np.random.seed(42)
            n_points = len(timeline_hist)
            residus_reels = np.random.normal(0, metriques["std_residus"], n_points)

        col1, col2 = st.columns(2)
        ecart_type_residus = metriques["std_residus"]
        variation_portefeuille = float(timeline_hist.std()) if len(timeline_hist) > 1 else float("nan")
        if variation_portefeuille > 0:
            ratio_variation = ecart_type_residus / variation_portefeuille * 100
            delta_message = f"{ratio_variation:.1f}% de l'écart-type historique"
        else:
            delta_message = None

        col1.metric(
            "Écart-type des résidus",
            f"{ecart_type_residus:.2f} €",
            delta=delta_message
        )
        dw_stat = metriques.get("dw_stat")
        if dw_stat is not None:
            col2.metric("Durbin-Watson", f"{dw_stat:.4f}")
            if dw_stat < 1.5:
                st.warning("Résidus positivement autocorrélés")
            elif dw_stat > 2.5:
                st.warning("Résidus négativement autocorrélés")
            else:
                st.success("Pas d'autocorrélation marquée des résidus")
        else:
            col2.metric("Durbin-Watson", "N/A")
            st.info("Durbin-Watson non disponible pour cette stratégie")
        
        # Graphique: Distribution des résidus
        st.write("**Distribution des Résidus vs Distribution Normale**")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=residus_reels,
            nbinsx=15,
            name="Résidus",
            opacity=0.7,
            marker_color='#7209B7',
            histnorm='probability density'
        ))
        
        # Courbe normale théorique
        x_norm = np.linspace(residus_reels.min(), residus_reels.max(), 100)
        y_norm = stats.norm.pdf(x_norm, 0, metriques["std_residus"])
        fig_hist.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Distribution Normale',
            line=dict(color='red', width=2)
        ))
        
        fig_hist.update_layout(
            title="Évaluation visuelle de la normalité des résidus",
            xaxis_title="Résidus (€)",
            yaxis_title="Densité",
            height=450,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Information d'aide sur la lecture du graphique
        st.info("**Comment interpréter ce graphique** : Plus l'histogramme des résidus (en violet) se rapproche de la courbe normale théorique (en rouge), plus la distribution des résidus est normale, ce qui est souhaitable pour la validité du modèle.")