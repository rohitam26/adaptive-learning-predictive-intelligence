"""
Integration and End-to-End API Tests for FastAPI backend.
"""

from fastapi.testclient import TestClient

from backend.app.main import app

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
