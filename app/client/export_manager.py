from __future__ import annotations
from typing import Dict
from datetime import datetime
import io
import base64

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import xlsxwriter
import numpy as np

from ui_manager import InterfaceUtilisateur

class GestionnaireExport:

    def afficher_options_export(
            self, 
            donnees_portefeuille: Dict, 
            donnees_prediction: Dict, 
            parametres: Dict,
            serie_benchmark: pd.Series = None
        ) -> None:
            """
            Affiche les options d'export des résultats.
            """
            st.subheader("📁 Export des résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    fichier_excel = self._generer_excel(donnees_portefeuille, donnees_prediction, parametres)
                    st.download_button(
                        label="📊 Export Excel",
                        data=fichier_excel,
                        file_name=f"optinvest_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excel_download",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erreur Excel: {e}")
        
            with col2:
                try:
                    fichier_pdf = self._generer_pdf(donnees_portefeuille, donnees_prediction, parametres, serie_benchmark)
                    st.download_button(
                        label="📄 Export PDF",
                        data=fichier_pdf,
                        file_name=f"optinvest_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="pdf_download",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erreur PDF: {e}")

    def _generer_excel(
        self, 
        donnees_portefeuille: Dict, 
        donnees_prediction: Dict, 
        parametres: Dict
    ) -> bytes:
        """
        Génère un fichier Excel complet avec tous les résultats.
        """
        buffer_sortie = io.BytesIO()
        
        workbook = xlsxwriter.Workbook(buffer_sortie, {'in_memory': True})
        
        format_titre = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'bg_color': '#FF6B35',
            'font_color': 'white'
        })
        
        format_entete = workbook.add_format({
            'bold': True,
            'bg_color': '#E6E6E6',
            'border': 1
        })
        
        format_nombre = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })
        
        format_pourcentage = workbook.add_format({
            'num_format': '0.00%',
            'border': 1
        })

        self._ecrire_onglet_resume(workbook, donnees_portefeuille, parametres, format_titre, format_entete, format_nombre, format_pourcentage)
        self._ecrire_onglet_timeline(workbook, donnees_portefeuille, format_entete, format_nombre)
        self._ecrire_onglet_predictions(workbook, donnees_prediction, format_entete, format_nombre)
        self._ecrire_onglet_residus(workbook, donnees_prediction, format_entete, format_nombre) 
        self._ecrire_onglet_metriques_detaillees(workbook, donnees_prediction, format_entete, format_nombre)
        self._ecrire_onglet_parametres(workbook, parametres, format_entete, format_nombre)
        
        workbook.close()
        return buffer_sortie.getvalue()

    def _ecrire_onglet_resume(self, workbook, donnees_portefeuille, parametres, format_titre, format_entete, format_nombre, format_pourcentage):
        """
        Écrit l'onglet résumé dans Excel.
        """
        interface = InterfaceUtilisateur()
        
        worksheet = workbook.add_worksheet('Résumé')
        
        worksheet.write('A1', 'OPTINVEST - Rapport de Simulation', format_titre)
        worksheet.merge_range('A1:D1', 'OPTINVEST - Rapport de Simulation', format_titre)
        
        cash_total = interface.calculer_cash_investi(
            parametres["duree_simulation_annees"],
            parametres["frequence_investissement"], 
            parametres["montant_initial"],
            parametres["apport_periodique"]
        )
        
        metriques = [
            ['Métrique', 'Valeur'],
            ['CAGR', donnees_portefeuille['cagr']],
            ['Rendement Total (€)', donnees_portefeuille['rendement_total']],
            ['Cash Investi (€)', cash_total],
            ['Valeur Finale (€)', cash_total + donnees_portefeuille['rendement_total']]
        ]
        
        for row_num, row_data in enumerate(metriques, 3):
            for col_num, cell_data in enumerate(row_data):
                if row_num == 3:
                    worksheet.write(row_num, col_num, cell_data, format_entete)
                elif col_num == 1 and row_num == 4:
                    worksheet.write(row_num, col_num, cell_data, format_pourcentage)
                elif col_num == 1:
                    worksheet.write(row_num, col_num, cell_data, format_nombre)
                else:
                    worksheet.write(row_num, col_num, cell_data)

    def _ecrire_onglet_timeline(self, workbook, donnees_portefeuille, format_entete, format_nombre):
        """
        Écrit l'onglet timeline dans Excel.
        """
        worksheet = workbook.add_worksheet('Timeline')
        
        timeline_data = donnees_portefeuille["timeline"]
        
        worksheet.write('A1', 'Date', format_entete)
        worksheet.write('B1', 'Valeur_Portefeuille', format_entete)
        
        row = 1
        for date_str, valeur in timeline_data.items():
            worksheet.write(row, 0, date_str)
            worksheet.write(row, 1, valeur, format_nombre)
            row += 1

    def _ecrire_onglet_predictions(self, workbook, donnees_prediction, format_entete, format_nombre):
        """
        Écrit l'onglet prédictions dans Excel.
        """
        worksheet = workbook.add_worksheet('Prédictions')
        
        entetes = ['Stratégie', 'Date', 'Valeur_Prédite', 'R2', 'RMSE']
        for col, entete in enumerate(entetes):
            worksheet.write(0, col, entete, format_entete)
        
        row = 1
        for nom_strategie, donnees_strategie in donnees_prediction.get("strategies", {}).items():
            for date_str, valeur in donnees_strategie["predictions"].items():
                worksheet.write(row, 0, nom_strategie)
                worksheet.write(row, 1, date_str)
                worksheet.write(row, 2, valeur, format_nombre)
                worksheet.write(row, 3, donnees_strategie["regression_metrics"]["r2"], format_nombre)
                worksheet.write(row, 4, donnees_strategie["regression_metrics"]["rmse"], format_nombre)
                row += 1

    def _ecrire_onglet_parametres(self, workbook, parametres, format_entete, format_nombre):
        """
        Écrit l'onglet paramètres dans Excel.
        """
        worksheet = workbook.add_worksheet('Paramètres')
        
        worksheet.write('A1', 'Paramètre', format_entete)
        worksheet.write('B1', 'Valeur', format_entete)
        
        parametres_liste = [
            ["Actifs", ", ".join(parametres["base"]["actifs"])],
            ["Durée (années)", parametres["duree_simulation_annees"]],
            ["Montant Initial (€)", parametres["montant_initial"]],
            ["Apport Périodique (€)", parametres["apport_periodique"]],
            ["Fréquence", parametres["frequence_investissement"]],
            ["Frais de Gestion (%)", parametres["base"]["frais_gestion"]],
            ["Durée Prédiction (années)", parametres["duree_prediction_annees"]]
        ]
        
        for row, (param, valeur) in enumerate(parametres_liste, 1):
            worksheet.write(row, 0, param)
            if isinstance(valeur, (int, float)):
                worksheet.write(row, 1, valeur, format_nombre)
            else:
                worksheet.write(row, 1, valeur)

    def _ecrire_onglet_residus(self, workbook, donnees_prediction, format_entete, format_nombre):
        """
        Écrit l'onglet résidus dans Excel.
        """
        worksheet = workbook.add_worksheet('Résidus')
        
        entetes = ['Stratégie', 'Index_Point', 'Valeur_Résidu', 'Résidu_Absolu', 'Résidu_Carré']
        for col, entete in enumerate(entetes):
            worksheet.write(0, col, entete, format_entete)
        
        row = 1
        for nom_strategie, donnees_strategie in donnees_prediction.get("strategies", {}).items():
            if "residus" in donnees_strategie and donnees_strategie["residus"]:
                residus = np.array(donnees_strategie["residus"])
                
                for idx, residu in enumerate(residus):
                    worksheet.write(row, 0, nom_strategie)
                    worksheet.write(row, 1, idx + 1)
                    worksheet.write(row, 2, residu, format_nombre)
                    worksheet.write(row, 3, abs(residu), format_nombre)
                    worksheet.write(row, 4, residu**2, format_nombre)
                    row += 1

    def _ecrire_onglet_metriques_detaillees(self, workbook, donnees_prediction, format_entete, format_nombre):
        """
        Écrit l'onglet métriques détaillées dans Excel.
        """
        worksheet = workbook.add_worksheet('Métriques Détaillées')
        
        entetes = ['Stratégie', 'R²', 'RMSE', 'MAE', 'Std_Résidus', 'Pente', 'Intercept', 
                  'Test_Normalité_P_Value']
        
        for col, entete in enumerate(entetes):
            worksheet.write(0, col, entete, format_entete)
        
        row = 1
        for nom_strategie, donnees_strategie in donnees_prediction.get("strategies", {}).items():
            metriques = donnees_strategie["regression_metrics"]
            
            worksheet.write(row, 0, nom_strategie)
            worksheet.write(row, 1, metriques.get("r2", 0), format_nombre)
            worksheet.write(row, 2, metriques.get("rmse", 0), format_nombre)
            worksheet.write(row, 3, metriques.get("mae", 0), format_nombre)
            worksheet.write(row, 4, metriques.get("std_residus", 0), format_nombre)
            worksheet.write(row, 5, metriques.get("pente", 0), format_nombre)
            worksheet.write(row, 6, metriques.get("intercept", 0), format_nombre)

    def _creer_graphique_residus_distribution(self, donnees_strategie: Dict, nom_strategie: str) -> bytes:
        """
        Crée le graphique de distribution des résidus.
        """
        metriques = donnees_strategie["regression_metrics"]
        
        if "residus" in donnees_strategie and donnees_strategie["residus"]:
            residus_reels = np.array(donnees_strategie["residus"])
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=residus_reels,
            nbinsx=15,
            name="Résidus",
            opacity=0.7,
            marker_color='#7209B7',
            histnorm='probability density'
        ))
        
        from scipy import stats
        x_norm = np.linspace(residus_reels.min(), residus_reels.max(), 100)
        y_norm = stats.norm.pdf(x_norm, 0, metriques["std_residus"])
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Distribution Normale',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Distribution des Résidus - {nom_strategie}",
            xaxis_title="Résidus (€)",
            yaxis_title="Densité",
            width=600,
            height=400,
            template="plotly_white"
        )
        
        img_bytes = pio.to_image(fig, format="png", width=600, height=400)
        return img_bytes

    def _creer_graphique_portefeuille(self, donnees_portefeuille: Dict, parametres: Dict) -> bytes:
        """
        Crée le graphique principal du portefeuille et retourne l'image au format bytes.
        """
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        dates_cash = pd.date_range(timeline_portefeuille.index.min(), timeline_portefeuille.index.max(), freq="ME")
        cash_cumule = parametres["montant_initial"] + np.arange(len(dates_cash)) * parametres["apport_periodique"]
        serie_cash = pd.Series(cash_cumule, index=dates_cash)
        
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
            title="Valeur du portefeuille vs cash investi",
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            width=600,
            height=400,
            template="plotly_white"
        )
        
        img_bytes = pio.to_image(fig, format="png", width=600, height=400)
        return img_bytes

    def _creer_graphique_predictions(self, donnees_prediction: Dict) -> bytes:
        """
        Crée le graphique des prédictions DCA et retourne l'image au format bytes.
        """
        couleurs = ['#FF6B35', '#004E89', '#009639', '#7209B7', '#D62828', '#F77F00']
        
        fig = go.Figure()
        
        for index_strategie, (nom_strategie, donnees_strategie) in enumerate(donnees_prediction.get("strategies", {}).items()):
            couleur_strategie = couleurs[index_strategie % len(couleurs)]
            
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
            title="Comparaison des Stratégies d'Investissement",
            xaxis_title="Date",
            yaxis_title="Valeur du Portefeuille (€)",
            width=600,
            height=400,
            template="plotly_white"
        )
        
        img_bytes = pio.to_image(fig, format="png", width=600, height=400)
        return img_bytes
    
    def _creer_graphique_benchmark(self, donnees_portefeuille: Dict, serie_benchmark: pd.Series, parametres: Dict) -> bytes:
        """
        Crée le graphique de comparaison avec le benchmark.
        """
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_portefeuille.index,
            y=timeline_portefeuille.values,
            mode='lines',
            name='Portefeuille',
            line=dict(color='#FF6B35', width=3)
        ))
        
        if not serie_benchmark.empty:
            fig.add_trace(go.Scatter(
                x=serie_benchmark.index,
                y=serie_benchmark.values,
                mode='lines',
                name='Indice ACWI IMI',
                line=dict(color='#009639', width=3)
            ))
        
        fig.update_layout(
            title="Comparaison avec l'indice ACWI IMI",
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            width=600,
            height=400,
            template="plotly_white"
        )
        
        img_bytes = pio.to_image(fig, format="png", width=600, height=400)
        return img_bytes

    def _creer_graphique_rendements(self, donnees_portefeuille: Dict) -> bytes:
        """
        Crée le graphique de distribution des rendements mensuels.
        """
        timeline_portefeuille = pd.Series(donnees_portefeuille["timeline"], dtype=float)
        timeline_portefeuille.index = pd.to_datetime(timeline_portefeuille.index)
        
        rendements_mensuels = timeline_portefeuille.resample("ME").last().pct_change().dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rendements_mensuels.index,
            y=np.array(rendements_mensuels.values) * 100,
            name='Rendements mensuels',
            marker_color='#7209B7'
        ))
        fig.update_layout(
            title="Distribution des rendements mensuels",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            width=600,
            height=400,
            template="plotly_white"
        )
        
        img_bytes = pio.to_image(fig, format="png", width=600, height=400)
        return img_bytes

    def _generer_pdf(
        self, 
        donnees_portefeuille: Dict, 
        donnees_prediction: Dict, 
        parametres: Dict,
        serie_benchmark: pd.Series = None
    ) -> bytes:
        """
        Génère un rapport PDF complet avec graphiques.
        """
        buffer_sortie = io.BytesIO()
        
        doc = SimpleDocTemplate(buffer_sortie, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        titre_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#FF6B35')
        )
        
        story.append(Paragraph("OPTINVEST - Rapport de Simulation", titre_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        self._ajouter_section_resume(story, styles, donnees_portefeuille, parametres)
        self._ajouter_section_parametres(story, styles, parametres)
        
        self._ajouter_graphiques_pdf(story, styles, donnees_portefeuille, donnees_prediction, parametres, serie_benchmark)
        
        self._ajouter_section_predictions(story, styles, donnees_prediction)
        
        doc.build(story)
        return buffer_sortie.getvalue()

    def _ajouter_graphiques_pdf(self, story, styles, donnees_portefeuille, donnees_prediction, parametres, serie_benchmark):
        """
        Ajoute tous les graphiques au PDF.
        """
        story.append(PageBreak())
        story.append(Paragraph("Graphiques d'Analyse", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        try:
            story.append(Paragraph("1. Évolution du portefeuille vs cash investi", styles['Heading3']))
            img_portefeuille = self._creer_graphique_portefeuille(donnees_portefeuille, parametres)
            buffer_img = io.BytesIO(img_portefeuille)
            img = Image(buffer_img, width=5*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
            if serie_benchmark is not None and not serie_benchmark.empty:
                story.append(Paragraph("2. Comparaison avec l'indice ACWI IMI", styles['Heading3']))
                img_benchmark = self._creer_graphique_benchmark(donnees_portefeuille, serie_benchmark, parametres)
                buffer_img_bench = io.BytesIO(img_benchmark)
                img_bench = Image(buffer_img_bench, width=5*inch, height=3.5*inch)
                story.append(img_bench)
                story.append(Spacer(1, 20))
            
            story.append(Paragraph("3. Distribution des rendements mensuels", styles['Heading3']))
            img_rendements = self._creer_graphique_rendements(donnees_portefeuille)
            buffer_img_rend = io.BytesIO(img_rendements)
            img_rend = Image(buffer_img_rend, width=5*inch, height=3.5*inch)
            story.append(img_rend)
            story.append(Spacer(1, 20))
            
            if "strategies" in donnees_prediction:
                story.append(PageBreak())
                story.append(Paragraph("4. Comparaison des stratégies d'investissement", styles['Heading3']))
                img_predictions = self._creer_graphique_predictions(donnees_prediction)
                buffer_img_pred = io.BytesIO(img_predictions)
                img_pred = Image(buffer_img_pred, width=5*inch, height=3.5*inch)
                story.append(img_pred)
                story.append(Spacer(1, 20))
                
                story.append(Paragraph("5. Analyse des Résidus par Stratégie", styles['Heading3']))
                story.append(Spacer(1, 10))
                
                for nom_strategie, donnees_strategie in donnees_prediction["strategies"].items():
                    story.append(Paragraph(f"Stratégie: {nom_strategie}", styles['Heading4']))
                    
                    story.append(Paragraph("Distribution des résidus", styles['Normal']))
                    img_dist = self._creer_graphique_residus_distribution(donnees_strategie, nom_strategie)
                    buffer_img_dist = io.BytesIO(img_dist)
                    img_dist_obj = Image(buffer_img_dist, width=4*inch, height=3*inch)
                    story.append(img_dist_obj)
                    story.append(Spacer(1, 10))
                
        except Exception as e:
            story.append(Paragraph(f"Erreur lors de la génération des graphiques: {str(e)}", styles['Normal']))

    def _ajouter_section_resume(self, story, styles, donnees_portefeuille, parametres):
        """
        Ajoute la section résumé au PDF.
        """
        interface = InterfaceUtilisateur()
        
        story.append(Paragraph("Résumé de Performance", styles['Heading2']))
        
        cash_total = interface.calculer_cash_investi(
            parametres["duree_simulation_annees"],
            parametres["frequence_investissement"],
            parametres["montant_initial"], 
            parametres["apport_periodique"]
        )
        
        donnees_tableau = [
            ['Métrique', 'Valeur'],
            ['CAGR', f"{donnees_portefeuille['cagr']*100:.2f}%"],
            ['Rendement Total', f"{donnees_portefeuille['rendement_total']:,.2f} €"],
            ['Cash Investi', f"{cash_total:,.2f} €"],
            ['Valeur Finale', f"{cash_total + donnees_portefeuille['rendement_total']:,.2f} €"]
        ]
        
        tableau = Table(donnees_tableau)
        tableau.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B35')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(tableau)
        story.append(Spacer(1, 20))

    def _ajouter_section_parametres(self, story, styles, parametres):
        """
        Ajoute la section paramètres au PDF.
        """
        story.append(Paragraph("Paramètres de Simulation", styles['Heading2']))
        
        params_text = f"""
        • Actifs sélectionnés: {', '.join(parametres['base']['actifs'])}
        • Durée de simulation: {parametres['duree_simulation_annees']} années
        • Montant initial: {parametres['montant_initial']:,.2f} €
        • Apport périodique: {parametres['apport_periodique']:,.2f} €
        • Fréquence d'investissement: {parametres['frequence_investissement']}
        • Frais de gestion: {parametres['base']['frais_gestion']:.1f}%
        • Durée de prédiction: {parametres['duree_prediction_annees']} années
        """
        
        story.append(Paragraph(params_text, styles['Normal']))
        story.append(Spacer(1, 20))

    def _ajouter_section_predictions(self, story, styles, donnees_prediction):
        """
        Ajoute la section prédictions au PDF avec plus de détails.
        """
        if "strategies" in donnees_prediction:
            story.append(PageBreak())
            story.append(Paragraph("Analyse Détaillée des Prédictions", styles['Heading2']))
            
            for nom_strategie, donnees_strategie in donnees_prediction["strategies"].items():
                metriques = donnees_strategie["regression_metrics"]
                
                story.append(Paragraph(f"Stratégie: {nom_strategie}", styles['Heading3']))
                
                texte_metriques = f"""
                <b>Métriques de Régression:</b><br/>
                • Coefficient R²: {metriques['r2']:.4f}<br/>
                • RMSE: {metriques['rmse']:.2f} €<br/>
                • MAE: {metriques.get('mae', 'N/A'):.2f} €<br/>
                • Écart-type des résidus: {metriques['std_residus']:.2f} €<br/>
                • Pente: {metriques.get('pente', 'N/A'):.4f}<br/>
                • Intercept: {metriques.get('intercept', 'N/A'):.2f}<br/>
                """
                
                story.append(Paragraph(texte_metriques, styles['Normal']))
                story.append(Spacer(1, 15))