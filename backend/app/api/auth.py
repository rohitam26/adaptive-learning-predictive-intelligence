"""
Authentication Endpoints: Register, Login, Seed Demo Accounts.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from backend.app.database.session import get_db
from backend.app.models.entities import User
from backend.app.schemas.schemas import Token, UserLogin, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=Token)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check existing user
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    hashed_pwd = get_password_hash(user_in.password)
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_pwd,
        full_name=user_in.full_name,
        role=user_in.role or "student",
        student_id=user_in.student_id or 100001,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    token = create_access_token(
        subject=db_user.email, role=db_user.role, student_id=db_user.student_id
    )
    return Token(
        access_token=token,
        token_type="bearer",
        role=db_user.role,
        student_id=db_user.student_id,
        full_name=db_user.full_name,
    )


@router.post("/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_in.email).first()
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password credentials.",
        )

    token = create_access_token(subject=user.email, role=user.role, student_id=user.student_id)
    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        student_id=user.student_id,
        full_name=user.full_name,
    )
