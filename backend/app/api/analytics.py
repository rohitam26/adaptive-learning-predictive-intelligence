"""
Cohort Analytics, Resource Intelligence and Model Status Endpoints.
"""

from fastapi import APIRouter, Depends

from backend.app.core.security import TokenPayload, get_current_user_token
from backend.app.schemas.schemas import (
    CohortAnalyticsResponse,
    ModelStatusResponse,
    ResourceAnalyticsResponse,
)
from backend.app.services.ml_service import MLService

router = APIRouter(tags=["Analytics"])


@router.get("/analytics/cohort", response_model=CohortAnalyticsResponse)
def get_cohort_analytics(current_user: TokenPayload = Depends(get_current_user_token)):
    ml_service = MLService.get_instance()
    return CohortAnalyticsResponse(**ml_service.get_cohort_analytics())


@router.get("/analytics/resources", response_model=ResourceAnalyticsResponse)
def get_resource_analytics(current_user: TokenPayload = Depends(get_current_user_token)):
    ml_service = MLService.get_instance()
    return ResourceAnalyticsResponse(**ml_service.get_resource_analytics())


@router.get("/models/status", response_model=ModelStatusResponse)
def get_models_status(current_user: TokenPayload = Depends(get_current_user_token)):
    ml_service = MLService.get_instance()
    return ModelStatusResponse(**ml_service.get_models_status())
