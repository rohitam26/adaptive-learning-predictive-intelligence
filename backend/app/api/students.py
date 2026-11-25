"""
Student Analytics and Intelligence Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import TokenPayload, get_current_user_token
from backend.app.database.session import get_db
from backend.app.models.entities import User
from backend.app.schemas.schemas import (
    AcademicRiskResponse,
    EngagementAnalyticsResponse,
    PerformanceMetricsResponse,
    StudentProfileResponse,
    StudentRecommendationsResponse,
    UserResponse,
)
from backend.app.services.ml_service import MLService

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: TokenPayload = Depends(get_current_user_token), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == current_user.sub).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        student_id=user.student_id,
    )


@router.get("/{student_id}/profile", response_model=StudentProfileResponse)
def get_student_profile(
    student_id: int, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    profile = ml_service.get_student_profile(student_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student record with ID {student_id} not found.",
        )
    return StudentProfileResponse(**profile)


@router.get("/{student_id}/performance", response_model=PerformanceMetricsResponse)
def get_student_performance(
    student_id: int, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    perf = ml_service.get_student_performance(student_id)
    return PerformanceMetricsResponse(**perf)


@router.get("/{student_id}/risk", response_model=AcademicRiskResponse)
def get_student_risk(student_id: int, current_user: TokenPayload = Depends(get_current_user_token)):
    ml_service = MLService.get_instance()
    risk = ml_service.get_student_risk(student_id)
    return AcademicRiskResponse(**risk)


@router.get("/{student_id}/recommendations", response_model=StudentRecommendationsResponse)
def get_student_recommendations(
    student_id: int, top_n: int = 6, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    recs = ml_service.get_student_recommendations(student_id, top_n=top_n)
    return StudentRecommendationsResponse(**recs)


@router.get("/{student_id}/engagement", response_model=EngagementAnalyticsResponse)
def get_student_engagement(
    student_id: int, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    eng = ml_service.get_student_engagement(student_id)
    return EngagementAnalyticsResponse(**eng)
