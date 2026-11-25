"""
Pydantic Schemas for API Requests and Responses.
"""

from typing import Any

from pydantic import BaseModel, EmailStr


# Auth
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str | None = "student"
    student_id: int | None = 100001


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    student_id: int | None
    full_name: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    student_id: int | None


# Student Schemas
class StudentProfileResponse(BaseModel):
    student_id: int
    modules: list[str]
    gender: str | None
    region: str | None
    highest_education: str | None
    imd_band: str | None
    age_band: str | None
    num_of_prev_attempts: int
    studied_credits: int
    disability: str
    final_result: str | None


class PerformanceMetricsResponse(BaseModel):
    student_id: int
    assessments_completed: int
    mean_score: float
    min_score: float
    max_score: float
    late_submissions: int
    mean_delay_days: float
    assessment_history: list[dict[str, Any]]


class ShapFactor(BaseModel):
    feature: str
    display_name: str
    feature_value: float
    shap_value: float
    impact: str
    magnitude: float


class AcademicRiskResponse(BaseModel):
    student_id: int
    model_name: str
    risk_probability: float
    risk_tier: str  # Low, Elevated, Critical
    confidence: float
    top_contributing_factors: list[ShapFactor]
    limitations: list[str]
    assessed_at: str


class ResourceRecommendation(BaseModel):
    resource_id: int
    module: str
    activity_type: str
    affinity_score: float
    graph_affinity: float
    popularity_score: float
    has_interacted: bool
    rationale: str


class StudentRecommendationsResponse(BaseModel):
    student_id: int
    total_recommendations: int
    recommendations: list[ResourceRecommendation]


class WeeklyEngagement(BaseModel):
    week: int
    clicks: int


class EngagementAnalyticsResponse(BaseModel):
    student_id: int
    total_clicks: int
    unique_resources_accessed: int
    active_days: int
    clicks_per_active_day: float
    clicks_by_activity_type: dict[str, int]
    weekly_trend: list[WeeklyEngagement]


# Cohort & Platform Analytics
class RiskDistribution(BaseModel):
    low: int
    elevated: int
    critical: int


class CohortAnalyticsResponse(BaseModel):
    total_students: int
    at_risk_count: int
    at_risk_percentage: float
    risk_distribution: RiskDistribution
    average_engagement_clicks: float
    average_assessment_score: float
    module_breakdown: list[dict[str, Any]]
    global_feature_importance: list[dict[str, Any]]


class ResourceAnalyticsResponse(BaseModel):
    total_resources: int
    activity_type_counts: dict[str, int]
    most_popular_resources: list[dict[str, Any]]


class ModelStatusResponse(BaseModel):
    champion_risk_model: dict[str, Any]
    ml_models: dict[str, Any]
    deep_learning_models: dict[str, Any]
    graph_learning: dict[str, Any]
    active_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database: str
    artifacts_ready: bool
