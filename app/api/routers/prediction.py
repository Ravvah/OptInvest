from fastapi import APIRouter, HTTPException, status
import traceback
from loguru import logger


from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.predictor import LinearModelPredictor

router = APIRouter(
    prefix="/predire",
    tags=["prediction"],
)

@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire et comparer les stratégies DCA",
)
async def predict_portfolio_future_growth(parametres_requete: PredictionRequest) -> PredictionResponse:
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
    
    # date_debut_simulation = parametres_requete.date_debut or (
    #     date.today() - timedelta(days=365 * parametres_requete.duree_ans)
    # )
    # date_fin_simulation = date_debut_simulation.replace(
    #     year=date_debut_simulation.year + parametres_requete.duree_ans
    # )
    
    try:
        predicteur_portefeuille = LinearModelPredictor(parametres_requete)
        logger.info("------ GET PREDICTION RESPONSE -------")
        resultats = predicteur_portefeuille._get_prediction_response()
        import math
        for k, v in resultats.model_dump().items():
            if isinstance(v, float) and not math.isfinite(v):
                print(f"Champ non valide : {k} = {v}")
        return resultats
        
    except Exception as erreur:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(erreur)}"
        )