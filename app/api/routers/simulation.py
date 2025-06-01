from datetime import date, timedelta
from fastapi import APIRouter, status

from app.api.schemas.simulation import SimulationRequest, SimulationResponse
from app.core.simulator import PortefeuilleSimulator

router = APIRouter(
    prefix="/simuler",
    tags=["simulation"],
)

simulateur_portefeuille = PortefeuilleSimulator()


@router.post(
    "/",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Simuler un portefeuille d'investissement",
)
async def simuler_portefeuille(parametres_requete: SimulationRequest) -> SimulationResponse:
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

    date_debut_simulation = parametres_requete.date_debut or (
        date.today() - timedelta(days=365 * parametres_requete.duree_ans)
    )
    date_fin_simulation = date_debut_simulation.replace(
        year=date_debut_simulation.year + parametres_requete.duree_ans
    )
    
    resultats_simulation = simulateur_portefeuille.simuler(
        actifs=parametres_requete.actifs,
        start=date_debut_simulation,
        end=date_fin_simulation,
        montant_initial=parametres_requete.montant_initial,
        apport_periodique=parametres_requete.apport_periodique,
        frequence=parametres_requete.frequence,
        frais_gestion_pct=parametres_requete.frais_gestion,
    )
    
    return SimulationResponse(**resultats_simulation)