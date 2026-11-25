"""
Integration and End-to-End API Tests for FastAPI backend.
"""

import uuid

from fastapi.testclient import TestClient

from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.models.entities import (
    ModelMetadataRecord,
    PredictionRecord,
    RecommendationRecord,
)

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_ready"] is True


def test_auth_login_and_access():
    # Login as demo student
    login_res = client.post(
        "/api/v1/auth/login", json={"email": "student@mintx.edu", "password": "Pass123!"}
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["role"] == "student"
    assert token_data["student_id"] == 100001

    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Test /students/me
    me_res = client.get("/api/v1/students/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "student@mintx.edu"

    # 2. Test /students/{student_id}/profile
    profile_res = client.get("/api/v1/students/100001/profile", headers=headers)
    assert profile_res.status_code == 200
    profile_data = profile_res.json()
    assert profile_data["student_id"] == 100001
    assert len(profile_data["modules"]) > 0

    # 3. Test /students/{student_id}/performance
    perf_res = client.get("/api/v1/students/100001/performance", headers=headers)
    assert perf_res.status_code == 200
    perf_data = perf_res.json()
    assert "mean_score" in perf_data

    # 4. Test /students/{student_id}/risk
    risk_res = client.get("/api/v1/students/100001/risk", headers=headers)
    assert risk_res.status_code == 200
    risk_data = risk_res.json()
    assert "risk_probability" in risk_data
    assert risk_data["risk_tier"] in ["Low", "Elevated", "Critical"]
    assert len(risk_data["top_contributing_factors"]) > 0

    # 5. Test /students/{student_id}/recommendations
    rec_res = client.get("/api/v1/students/100001/recommendations", headers=headers)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["total_recommendations"] > 0
    assert len(rec_data["recommendations"]) > 0
    assert "rationale" in rec_data["recommendations"][0]

    # 6. Test /students/{student_id}/engagement
    eng_res = client.get("/api/v1/students/100001/engagement", headers=headers)
    assert eng_res.status_code == 200
    eng_data = eng_res.json()
    assert "weekly_trend" in eng_data

    # 7. Test /analytics/cohort
    cohort_res = client.get("/api/v1/analytics/cohort", headers=headers)
    assert cohort_res.status_code == 200
    cohort_data = cohort_res.json()
    assert cohort_data["total_students"] > 0
    assert "risk_distribution" in cohort_data

    # 8. Test /analytics/resources
    res_res = client.get("/api/v1/analytics/resources", headers=headers)
    assert res_res.status_code == 200
    res_data = res_res.json()
    assert res_data["total_resources"] > 0

    # 9. Test /models/status
    model_res = client.get("/api/v1/models/status", headers=headers)
    assert model_res.status_code == 200
    model_data = model_res.json()
    assert "champion_risk_model" in model_data


def test_registration_and_persistence_flow():
    # 1. Register student without providing student_id
    uid = uuid.uuid4().hex[:8]
    email = f"new_student_{uid}@mintx.edu"
    reg_res = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "Password123!",
            "full_name": "Test Student New",
            "role": "student",
        },
    )
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert reg_data["role"] == "student"
    assert reg_data["student_id"] is not None
    assigned_id = reg_data["student_id"]
    token = reg_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Profile lookup for newly registered student (DB fallback verification)
    prof_res = client.get(f"/api/v1/students/{assigned_id}/profile", headers=headers)
    assert prof_res.status_code == 200
    prof_data = prof_res.json()
    assert prof_data["student_id"] == assigned_id
    assert len(prof_data["modules"]) > 0

    # 3. Register educator (must have student_id = None)
    edu_res = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"new_educator_{uid}@mintx.edu",
            "password": "Password123!",
            "full_name": "Test Instructor",
            "role": "educator",
        },
    )
    assert edu_res.status_code == 200
    edu_data = edu_res.json()
    assert edu_data["role"] == "educator"
    assert edu_data["student_id"] is None

    # 4. Persistence verification: trigger risk & recommendation
    risk_res = client.get(f"/api/v1/students/{assigned_id}/risk", headers=headers)
    assert risk_res.status_code == 200
    rec_res = client.get(f"/api/v1/students/{assigned_id}/recommendations", headers=headers)
    assert rec_res.status_code == 200

    # Check database records
    db = SessionLocal()
    try:
        pred_records = (
            db.query(PredictionRecord).filter(PredictionRecord.student_id == assigned_id).all()
        )
        assert len(pred_records) >= 1
        assert pred_records[0].predicted_risk_tier in ["Low", "Elevated", "Critical"]

        rec_records = (
            db.query(RecommendationRecord)
            .filter(RecommendationRecord.student_id == assigned_id)
            .all()
        )
        assert len(rec_records) >= 1

        models = db.query(ModelMetadataRecord).all()
        assert len(models) >= 1
    finally:
        db.close()


def test_admin_registration_forbidden():
    """Ensure public registration cannot create Administrator accounts."""
    res = client.post(
        "/api/v1/auth/register",
        json={
            "email": "rogue_admin@mintx.edu",
            "password": "Password123!",
            "full_name": "Rogue Admin",
            "role": "admin",
        },
    )
    assert res.status_code == 403
    assert "Public registration cannot create Administrator accounts" in res.json()["detail"]


def test_student_id_allocation_safety_and_duplicate_prevention():
    """Verify student ID allocation prevents duplicate IDs and safely links StudentProfile."""
    # Attempt to steal existing student_id 100001
    steal_res = client.post(
        "/api/v1/auth/register",
        json={
            "email": "impostor@mintx.edu",
            "password": "Password123!",
            "full_name": "Impostor Student",
            "role": "student",
            "student_id": 100001,
        },
    )
    assert steal_res.status_code == 400
    assert "already registered" in steal_res.json()["detail"]

    # Register two consecutive students without passing student_id
    uid = uuid.uuid4().hex[:8]
    res1 = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"seq_student_1_{uid}@mintx.edu",
            "password": "Password123!",
            "full_name": "Seq Student 1",
            "role": "student",
        },
    )
    assert res1.status_code == 200
    id1 = res1.json()["student_id"]

    res2 = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"seq_student_2_{uid}@mintx.edu",
            "password": "Password123!",
            "full_name": "Seq Student 2",
            "role": "student",
        },
    )
    assert res2.status_code == 200
    id2 = res2.json()["student_id"]

    # Must be distinct
    assert id1 != id2
    assert id2 > id1

    # Check StudentProfile linkage in database
    from backend.app.models.entities import StudentProfile

    db = SessionLocal()
    try:
        p1 = db.query(StudentProfile).filter(StudentProfile.id_student == id1).first()
        p2 = db.query(StudentProfile).filter(StudentProfile.id_student == id2).first()
        assert p1 is not None
        assert p2 is not None
        assert p1.id_student == id1
        assert p2.id_student == id2
    finally:
        db.close()


def test_persistence_no_unexpected_duplicates():
    """Confirm prediction and recommendation persistence does not create duplicate records."""
    login_res = client.post(
        "/api/v1/auth/login", json={"email": "student@mintx.edu", "password": "Pass123!"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    student_id = 100001

    db = SessionLocal()
    try:
        # Initial counts
        init_preds = (
            db.query(PredictionRecord).filter(PredictionRecord.student_id == student_id).count()
        )
        init_recs = (
            db.query(RecommendationRecord)
            .filter(RecommendationRecord.student_id == student_id)
            .count()
        )

        # Call risk and recommendations twice consecutively
        client.get(f"/api/v1/students/{student_id}/risk", headers=headers)
        client.get(f"/api/v1/students/{student_id}/risk", headers=headers)

        client.get(f"/api/v1/students/{student_id}/recommendations", headers=headers)
        client.get(f"/api/v1/students/{student_id}/recommendations", headers=headers)

        final_preds = (
            db.query(PredictionRecord).filter(PredictionRecord.student_id == student_id).count()
        )
        final_recs = (
            db.query(RecommendationRecord)
            .filter(RecommendationRecord.student_id == student_id)
            .count()
        )

        # The count should increase by at most 1 prediction event (or remain 1 if already present), never duplicate per call
        assert final_preds <= max(init_preds, 1) + 1
        # Recommendation count should equal the number of distinct recommended resources, not double
        assert final_recs <= max(init_recs, 6)
    finally:
        db.close()


def test_existing_demo_accounts_login():
    """Verify demo student, educator, and admin accounts authenticate successfully."""
    # 1. Demo Student
    s_res = client.post(
        "/api/v1/auth/login", json={"email": "student@mintx.edu", "password": "Pass123!"}
    )
    assert s_res.status_code == 200
    assert s_res.json()["role"] == "student"
    assert s_res.json()["student_id"] == 100001

    # 2. Demo Educator
    e_res = client.post(
        "/api/v1/auth/login", json={"email": "educator@mintx.edu", "password": "Pass123!"}
    )
    assert e_res.status_code == 200
    assert e_res.json()["role"] == "educator"
    assert e_res.json()["student_id"] is None

    # 3. Demo Admin
    a_res = client.post(
        "/api/v1/auth/login", json={"email": "admin@mintx.edu", "password": "Pass123!"}
    )
    assert a_res.status_code == 200
    assert a_res.json()["role"] == "admin"
    assert a_res.json()["student_id"] is None


def test_concurrent_registrations_allocate_unique_ids():
    """Run a genuinely concurrent registration test using multiple simultaneous requests.

    Verify every successful registration receives a distinct ID and no collisions occur.
    """
    import concurrent.futures

    from backend.app.models.entities import StudentProfile, User

    num_concurrent = 8
    uids = [uuid.uuid4().hex[:8] for _ in range(num_concurrent)]

    def register_worker(uid_val):
        payload = {
            "email": f"concurrent_student_{uid_val}@mintx.edu",
            "password": "Password123!",
            "full_name": f"Concurrent Student {uid_val}",
            "role": "student",
        }
        res = client.post("/api/v1/auth/register", json=payload)
        return res.status_code, res.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(register_worker, uid) for uid in uids]
        results = [f.result() for f in futures]

    # Every request must succeed
    status_codes = [r[0] for r in results]
    assert all(code == 200 for code in status_codes), f"Not all succeeded: {status_codes}"

    # Verify every registered student receives a distinct student_id
    student_ids = [r[1]["student_id"] for r in results]
    assert len(student_ids) == num_concurrent
    assert len(set(student_ids)) == num_concurrent, f"Duplicate student IDs found: {student_ids}"

    # Verify each ID is tracked in the database with linked StudentProfile
    db = SessionLocal()
    try:
        for sid in student_ids:
            u = db.query(User).filter(User.student_id == sid).first()
            p = db.query(StudentProfile).filter(StudentProfile.id_student == sid).first()
            assert u is not None, f"User missing for student_id {sid}"
            assert p is not None, f"StudentProfile missing for student_id {sid}"
            assert p.id_student == sid
    finally:
        db.close()
