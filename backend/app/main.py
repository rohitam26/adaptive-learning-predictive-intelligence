"""
FastAPI Application Entry Point for Adaptive Learning & Predictive Intelligence Platform.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.app.api import analytics, auth, students
from backend.app.core.config import settings
from backend.app.core.security import get_password_hash
from backend.app.database.session import SessionLocal, init_db
from backend.app.models.entities import User
from backend.app.schemas.schemas import HealthResponse


def seed_database():
    init_db()
    db: Session = SessionLocal()
    try:
        demo_accounts = [
            {
                "email": "student@mintx.edu",
                "password": "Pass123!",
                "full_name": "Demo Student",
                "role": "student",
                "student_id": 100001,
            },
            {
                "email": "educator@mintx.edu",
                "password": "Pass123!",
                "full_name": "Dr. Sarah Adams",
                "role": "educator",
                "student_id": None,
            },
            {
                "email": "admin@mintx.edu",
                "password": "Pass123!",
                "full_name": "Platform Administrator",
                "role": "admin",
                "student_id": None,
            },
        ]
        for acc in demo_accounts:
            existing = db.query(User).filter(User.email == acc["email"]).first()
            if not existing:
                user = User(
                    email=acc["email"],
                    hashed_password=get_password_hash(acc["password"]),
                    full_name=acc["full_name"],
                    role=acc["role"],
                    student_id=acc["student_id"],
                )
                db.add(user)
        db.commit()
    finally:
        db.close()


# Ensure tables and demo seeds exist immediately
seed_database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_database()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Adaptive Learning and Predictive Intelligence Platform for Educational Analytics and Personalized Resource Recommendation.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for local development across Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(students.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    artifacts_ready = (settings.ARTIFACTS_DIR / "models" / "champion_risk_model.joblib").exists()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        version=settings.PROJECT_VERSION,
        database="connected",
        artifacts_ready=artifacts_ready,
    )


# Serve built frontend static files if available

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
