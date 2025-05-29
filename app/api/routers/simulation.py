from fastapi import APIRouter, HTTPException, status

from app.api.schemas.simulation import SimulationRequest, SimulationResponse

router = APIRouter(
    prefix="/simuler",          
    tags=["simulation"],         
)

@router.post(
    "/",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Simuler un portefeuille d’investissement",
)
async def simuler_portefeuille(params: SimulationRequest) -> SimulationResponse:
    """
    Lance une **simulation historique** sur la période demandée.

    **Paramètres** :
    - **actifs** : liste d’actifs autorisés (AAPL, MSFT, VWCE, SXR8, AGGH, TLT).  
    - **duree_ans** : durée de la simulation en années (entier > 0).  
    - **montant_initial** : capital initial en euros.  
    - **apport_periodique** : montant de chaque versement périodique.  
    - **frequence** : mensuelle / trimestrielle / semestrielle / annuelle.  
    - **frais_gestion** : pourcentage annuel prélevé en fin d’année.

    **Réponse** :  
    - **cagr** : taux de croissance annuel composé.  
    - **rendement_total** : gain net en euros.  
    - **timeline** : dict « date ISO → valeur du portefeuille ».
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Moteur de simulation non encore disponible.",
    )
