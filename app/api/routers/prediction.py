from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, status

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.predictor import PortefeuillePredictorDCA

router = APIRouter(
    prefix="/predire",
    tags=["prediction"],
)

predicteur_portefeuille = PortefeuillePredictorDCA()


@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire et comparer les stratégies DCA",
)
async def predire_strategies_dca(parametres_requete: PredictionRequest) -> PredictionResponse:
    """
    Lance une simulation avec prédictions pour comparer les stratégies DCA.

    Paramètres :
    - actifs : liste d'actifs autorisés
    - duree_ans : durée de la simulation historique
    - duree_prediction_ans : durée des prédictions
    - montant_initial : capital initial en euros
    - apport_periodique : montant de chaque versement périodique
    - frais_gestion : pourcentage annuel prélevé en fin d'année

    Réponse :
    - strategies : dictionnaire avec les résultats pour chaque stratégie (DCA + Lump Sum)
    """
    
    date_debut_simulation = parametres_requete.date_debut or (
        date.today() - timedelta(days=365 * parametres_requete.duree_ans)
    )
    date_fin_simulation = date_debut_simulation.replace(
        year=date_debut_simulation.year + parametres_requete.duree_ans
    )
    
    try:
        resultats_comparaison = predicteur_portefeuille.comparer_strategies_dca(
            actifs=parametres_requete.actifs,
            start=date_debut_simulation,
            end=date_fin_simulation,
            montant_initial=parametres_requete.montant_initial,
            apport_periodique=parametres_requete.apport_periodique,
            frais_gestion_pct=parametres_requete.frais_gestion,
            duree_prediction_ans=parametres_requete.duree_prediction_ans
        )
        
        return PredictionResponse(**resultats_comparaison)
        
    except Exception as erreur:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(erreur)}"
        )