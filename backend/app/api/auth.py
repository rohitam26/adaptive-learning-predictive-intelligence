"""
Authentication Endpoints: Register, Login, Seed Demo Accounts.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from backend.app.database.session import get_db
from backend.app.models.entities import StudentProfile, User
from backend.app.schemas.schemas import Token, UserLogin, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _ensure_student_profile(db: Session, student_id: int):
    existing_profile = (
        db.query(StudentProfile).filter(StudentProfile.id_student == student_id).first()
    )
    if not existing_profile:
        new_profile = StudentProfile(
            id_student=student_id,
            code_module="AAA",
            code_presentation="2014J",
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
        db.add(new_profile)


@router.post("/register", response_model=Token)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check existing user email
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    role = (user_in.role or "student").strip().lower()

    # 1. Enforce admin registration restriction
    if role in ["admin", "administrator"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration cannot create Administrator accounts. Administrator accounts require authorized administrative provisioning.",
        )

    if role not in ["student", "educator"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid registration role '{user_in.role}'. Only 'student' and 'educator' roles are permitted for self-registration.",
        )

    hashed_pwd = get_password_hash(user_in.password)

    # 2. Allocate or validate student_id safely with concurrency retry loop
    if role == "student":
        if user_in.student_id:
            existing_user_with_id = (
                db.query(User).filter(User.student_id == user_in.student_id).first()
            )
            if existing_user_with_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Student ID {user_in.student_id} is already registered to an existing user.",
                )
            assigned_student_id = user_in.student_id
            _ensure_student_profile(db, assigned_student_id)

            db_user = User(
                email=user_in.email,
                hashed_password=hashed_pwd,
                full_name=user_in.full_name,
                role=role,
                student_id=assigned_student_id,
            )
            try:
                db.add(db_user)
                db.commit()
                db.refresh(db_user)
            except IntegrityError:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Student ID {user_in.student_id} or email is already registered.",
                )
        else:
            max_retries = 10
            db_user = None
            for attempt in range(max_retries):
                max_user_id = db.query(func.max(User.student_id)).scalar() or 100004
                max_profile_id = db.query(func.max(StudentProfile.id_student)).scalar() or 100004
                assigned_student_id = max(max_user_id, max_profile_id, 100004) + 1 + attempt

                while (
                    db.query(User).filter(User.student_id == assigned_student_id).first()
                    is not None
                    or db.query(StudentProfile)
                    .filter(StudentProfile.id_student == assigned_student_id)
                    .first()
                    is not None
                ):
                    assigned_student_id += 1

                _ensure_student_profile(db, assigned_student_id)

                candidate_user = User(
                    email=user_in.email,
                    hashed_password=hashed_pwd,
                    full_name=user_in.full_name,
                    role=role,
                    student_id=assigned_student_id,
                )
                try:
                    db.add(candidate_user)
                    db.commit()
                    db.refresh(candidate_user)
                    db_user = candidate_user
                    break
                except IntegrityError:
                    db.rollback()
                    if attempt == max_retries - 1:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail="Unable to allocate unique student ID due to concurrent registration conflict. Please retry.",
                        )
    else:
        db_user = User(
            email=user_in.email,
            hashed_password=hashed_pwd,
            full_name=user_in.full_name,
            role=role,
            student_id=None,
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
