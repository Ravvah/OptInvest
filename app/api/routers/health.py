from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health", summary="Health Check")
async def health_check() -> dict:
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "API is running"}

