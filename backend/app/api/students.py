"""
Student Analytics and Intelligence Endpoints.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.security import TokenPayload, get_current_user_token
from backend.app.database.session import get_db
from backend.app.models.entities import (
    PredictionRecord,
    RecommendationRecord,
    StudentProfile,
    User,
)
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
    student_id: int,
    current_user: TokenPayload = Depends(get_current_user_token),
    db: Session = Depends(get_db),
):
    ml_service = MLService.get_instance()
    profile = ml_service.get_student_profile(student_id)
    if profile:
        return StudentProfileResponse(**profile)

    # Fallback to relational database StudentProfile
    db_profiles = db.query(StudentProfile).filter(StudentProfile.id_student == student_id).all()
    if db_profiles:
        first_p = db_profiles[0]
        modules = [p.code_module for p in db_profiles if p.code_module] or ["AAA"]
        return StudentProfileResponse(
            student_id=student_id,
            modules=modules,
            gender=first_p.gender,
            region=first_p.region,
            highest_education=first_p.highest_education,
            imd_band=first_p.imd_band,
            age_band=first_p.age_band,
            num_of_prev_attempts=first_p.num_of_prev_attempts or 0,
            studied_credits=first_p.studied_credits or 60,
            disability=first_p.disability or "N",
            final_result=first_p.final_result or "In Progress",
        )

    # Fallback to User table record
    db_user = db.query(User).filter(User.student_id == student_id).first()
    if db_user:
        return StudentProfileResponse(
            student_id=student_id,
            modules=["AAA"],
            gender="Unknown",
            region="East Anglian Region",
            highest_education="A Level or Equivalent",
            imd_band="50-60%",
            age_band="0-35",
            num_of_prev_attempts=0,
            studied_credits=60,
            disability="N",
            final_result="In Progress",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Student record with ID {student_id} not found.",
    )


@router.get("/{student_id}/performance", response_model=PerformanceMetricsResponse)
def get_student_performance(
    student_id: int, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    perf = ml_service.get_student_performance(student_id)
    return PerformanceMetricsResponse(**perf)


@router.get("/{student_id}/risk", response_model=AcademicRiskResponse)
def get_student_risk(
    student_id: int,
    current_user: TokenPayload = Depends(get_current_user_token),
    db: Session = Depends(get_db),
):
    ml_service = MLService.get_instance()
    risk = ml_service.get_student_risk(student_id)

    # Persist prediction to database without duplicate accumulation
    try:
        raw_factors = risk.get("top_contributing_factors", [])
        factors_json = [f.dict() if hasattr(f, "dict") else f for f in raw_factors]
        risk_prob = float(risk["risk_probability"])
        risk_tier = str(risk["risk_tier"])
        conf = float(risk["confidence"])
        model_name = str(risk.get("model_name", "Gradient Boosting"))

        latest_pred = (
            db.query(PredictionRecord)
            .filter(
                PredictionRecord.student_id == student_id,
                PredictionRecord.model_name == model_name,
            )
            .order_by(PredictionRecord.id.desc())
            .first()
        )

        if (
            latest_pred is not None
            and latest_pred.predicted_risk_tier == risk_tier
            and abs(latest_pred.predicted_risk_probability - risk_prob) < 1e-4
        ):
            latest_pred.created_at = datetime.now(UTC)
            latest_pred.confidence = conf
            latest_pred.shap_contributions = factors_json
        else:
            pred_record = PredictionRecord(
                student_id=student_id,
                model_name=model_name,
                model_version="1.0.0",
                predicted_risk_probability=risk_prob,
                predicted_risk_tier=risk_tier,
                confidence=conf,
                shap_contributions=factors_json,
            )
            db.add(pred_record)
        db.commit()
    except SQLAlchemyError:
        db.rollback()

    return AcademicRiskResponse(**risk)


@router.get("/{student_id}/recommendations", response_model=StudentRecommendationsResponse)
def get_student_recommendations(
    student_id: int,
    top_n: int = 6,
    current_user: TokenPayload = Depends(get_current_user_token),
    db: Session = Depends(get_db),
):
    ml_service = MLService.get_instance()
    recs = ml_service.get_student_recommendations(student_id, top_n=top_n)

    # Persist delivered recommendations without duplicate accumulation per student-resource pair
    try:
        for r in recs.get("recommendations", []):
            res_id = int(r["resource_id"])
            act_type = str(r["activity_type"])
            aff_score = float(r["affinity_score"])
            rationale_text = str(r["rationale"])

            existing_rec = (
                db.query(RecommendationRecord)
                .filter(
                    RecommendationRecord.student_id == student_id,
                    RecommendationRecord.resource_id == res_id,
                )
                .first()
            )
            if existing_rec is not None:
                existing_rec.affinity_score = aff_score
                existing_rec.rationale = rationale_text
                existing_rec.delivered_at = datetime.now(UTC)
            else:
                rec_record = RecommendationRecord(
                    student_id=student_id,
                    resource_id=res_id,
                    activity_type=act_type,
                    affinity_score=aff_score,
                    rationale=rationale_text,
                )
                db.add(rec_record)
        db.commit()
    except SQLAlchemyError:
        db.rollback()

    return StudentRecommendationsResponse(**recs)


@router.get("/{student_id}/engagement", response_model=EngagementAnalyticsResponse)
def get_student_engagement(
    student_id: int, current_user: TokenPayload = Depends(get_current_user_token)
):
    ml_service = MLService.get_instance()
    eng = ml_service.get_student_engagement(student_id)
    return EngagementAnalyticsResponse(**eng)
