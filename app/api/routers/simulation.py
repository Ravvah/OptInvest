from datetime import date, timedelta
from fastapi import APIRouter, status
from loguru import logger

from app.api.schemas.simulation import SimulationRequest, SimulationResponse
from app.core.simulator import Simulator

router = APIRouter(
    prefix="/simuler",
    tags=["simulation"],
)


@router.post(
    "/",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Simuler un portefeuille d'investissement",
)
async def simulate_portfolio_growth(parametres_requete: SimulationRequest) -> SimulationResponse:
    """
    Lance une simulation historique sur la période demandée.

    Paramètres :
    - actifs : liste d'actifs autorisés (AAPL, MSFT, VWCE, SXR8, AGGH, TLT)
    - duree_ans : durée de la simulation en années (entier > 0)
    - montant_initial : capital initial en euros
    - apport_periodique : montant de chaque versement périodique
    - frequence : mensuelle / trimestrielle / semestrielle / annuelle
    - frais_gestion : pourcentage annuel prélevé en fin d'année

    Réponse :
    - cagr : taux de croissance annuel composé
    - rendement_total : gain net en euros
    - timeline : dict « date ISO → valeur du portefeuille »
    """
    if not parametres_requete.date_debut:
        parametres_requete.date_debut = date.today() - timedelta(days=365 * parametres_requete.duree_ans)

    parametres_requete.date_fin = parametres_requete.date_debut.replace(
        year=parametres_requete.date_debut.year + parametres_requete.duree_ans
    )
    simulateur_portefeuille = Simulator(requete=parametres_requete)
    logger.info("--------- GET SIMULATION RESPONSE ----------")
    resultats_simulation = simulateur_portefeuille.simulate_portfolio_growth()
    return resultats_simulation