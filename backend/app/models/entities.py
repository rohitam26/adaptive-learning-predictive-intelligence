"""
SQLAlchemy Relational Database Models.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from backend.app.database.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="student")  # student, educator, admin
    student_id = Column(Integer, nullable=True, index=True)  # links to OULAD student ID
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id_student = Column(Integer, primary_key=True, index=True)
    code_module = Column(String(20), primary_key=True, index=True)
    code_presentation = Column(String(20), primary_key=True, index=True)
    gender = Column(String(10), nullable=True)
    region = Column(String(100), nullable=True)
    highest_education = Column(String(100), nullable=True)
    imd_band = Column(String(50), nullable=True)
    age_band = Column(String(50), nullable=True)
    num_of_prev_attempts = Column(Integer, default=0)
    studied_credits = Column(Integer, default=60)
    disability = Column(String(10), default="N")
    final_result = Column(String(50), nullable=True)


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    code_module = Column(String(20), nullable=False, index=True)
    code_presentation = Column(String(20), nullable=False, index=True)
    module_presentation_length = Column(Integer, default=268)


class Assessment(Base):
    __tablename__ = "assessments"

    id_assessment = Column(Integer, primary_key=True, index=True)
    code_module = Column(String(20), nullable=False)
    code_presentation = Column(String(20), nullable=False)
    assessment_type = Column(String(20), nullable=False)  # TMA, CMA, Exam
    date = Column(Integer, nullable=True)
    weight = Column(Float, default=0.0)


class LearningResource(Base):
    __tablename__ = "learning_resources"

    id_site = Column(Integer, primary_key=True, index=True)
    code_module = Column(String(20), nullable=False)
    code_presentation = Column(String(20), nullable=False)
    activity_type = Column(String(50), nullable=False)
    week_from = Column(Integer, nullable=True)
    week_to = Column(Integer, nullable=True)


class LearnerInteraction(Base):
    __tablename__ = "learner_interactions"

    id = Column(Integer, primary_key=True, index=True)
    code_module = Column(String(20), nullable=False)
    code_presentation = Column(String(20), nullable=False)
    id_student = Column(Integer, index=True, nullable=False)
    id_site = Column(Integer, index=True, nullable=False)
    date = Column(Integer, nullable=False)
    sum_click = Column(Integer, default=1)


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, index=True, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), default="1.0.0")
    predicted_risk_probability = Column(Float, nullable=False)
    predicted_risk_tier = Column(String(50), nullable=False)  # Low, Elevated, Critical
    confidence = Column(Float, nullable=False)
    shap_contributions = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class RecommendationRecord(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, index=True, nullable=False)
    resource_id = Column(Integer, nullable=False)
    activity_type = Column(String(50), nullable=False)
    affinity_score = Column(Float, nullable=False)
    rationale = Column(Text, nullable=False)
    delivered_at = Column(DateTime, default=lambda: datetime.now(UTC))


class ModelMetadataRecord(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(100), nullable=False)  # ML, DL, Graph, Hybrid
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    status = Column(String(50), default="active")  # active, archived
    hyperparameters = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))
