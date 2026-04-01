from datetime import datetime

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import APP_VERSION
router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        timestamp=datetime.utcnow().replace(microsecond=0),
    )
