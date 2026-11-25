"""
Script to test and verify all 12 FastAPI REST endpoints individually.
"""

import uuid

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

results = []

# 1. GET /health
r1 = client.get("/health")
results.append(
    (
        "GET",
        "/health",
        r1.status_code,
        r1.status_code == 200,
        f"Status: {r1.json().get('status')}, Database: {r1.json().get('database')}, Ready: {r1.json().get('artifacts_ready')}",
    )
)

# 2. POST /api/v1/auth/login
r2 = client.post("/api/v1/auth/login", json={"email": "student@mintx.edu", "password": "Pass123!"})
token = r2.json().get("access_token") if r2.status_code == 200 else None
headers = {"Authorization": f"Bearer {token}"} if token else {}
results.append(
    (
        "POST",
        "/api/v1/auth/login",
        r2.status_code,
        r2.status_code == 200,
        f"Role: {r2.json().get('role')}, Student ID: {r2.json().get('student_id')}, Token type: {r2.json().get('token_type')}",
    )
)

# 3. POST /api/v1/auth/register (generate unique email for idempotent verification)
unique_email = f"student_{uuid.uuid4().hex[:8]}@mintx.edu"
reg_payload = {
    "email": unique_email,
    "password": "Password123!",
    "full_name": "Test Verify User",
    "role": "student",
}
r3 = client.post("/api/v1/auth/register", json=reg_payload)
results.append(
    (
        "POST",
        "/api/v1/auth/register",
        r3.status_code,
        r3.status_code in [200, 201],
        f"Created User: {r3.json().get('email')}, Role: {r3.json().get('role')}",
    )
)

# 4. GET /api/v1/students/me
r4 = client.get("/api/v1/students/me", headers=headers)
results.append(
    (
        "GET",
        "/api/v1/students/me",
        r4.status_code,
        r4.status_code == 200,
        f"Email: {r4.json().get('email')}, Role: {r4.json().get('role')}, Student ID: {r4.json().get('student_id')}",
    )
)

# 5. GET /api/v1/students/{id}/profile
r5 = client.get("/api/v1/students/100001/profile", headers=headers)
results.append(
    (
        "GET",
        "/api/v1/students/{id}/profile",
        r5.status_code,
        r5.status_code == 200,
        f"Modules: {r5.json().get('modules')}, Education: {r5.json().get('highest_education')}, Region: {r5.json().get('region')}",
    )
)

# 6. GET /api/v1/students/{id}/performance
r6 = client.get("/api/v1/students/100001/performance", headers=headers)
results.append(
    (
        "GET",
        "/api/v1/students/{id}/performance",
        r6.status_code,
        r6.status_code == 200,
        f"Assessments Completed: {r6.json().get('assessments_completed')}, Mean Score: {r6.json().get('mean_score')}",
    )
)

# 7. GET /api/v1/students/{id}/risk
r7 = client.get("/api/v1/students/100001/risk", headers=headers)
top_factors = len(r7.json().get("top_contributing_factors", []))
results.append(
    (
        "GET",
        "/api/v1/students/{id}/risk",
        r7.status_code,
        r7.status_code == 200,
        f"Model: {r7.json().get('model_name')}, Risk Probability: {r7.json().get('risk_probability')}, Tier: {r7.json().get('risk_tier')}, Factors: {top_factors}",
    )
)

# 8. GET /api/v1/students/{id}/recommendations
r8 = client.get("/api/v1/students/100001/recommendations", headers=headers)
recs_count = r8.json().get("total_recommendations", 0)
results.append(
    (
        "GET",
        "/api/v1/students/{id}/recommendations",
        r8.status_code,
        r8.status_code == 200,
        f"Total Recommendations: {recs_count}, Top Site: {r8.json().get('recommendations', [{}])[0].get('resource_id')}",
    )
)

# 9. GET /api/v1/students/{id}/engagement
r9 = client.get("/api/v1/students/100001/engagement", headers=headers)
trend_len = len(r9.json().get("weekly_trend", []))
results.append(
    (
        "GET",
        "/api/v1/students/{id}/engagement",
        r9.status_code,
        r9.status_code == 200,
        f"Weekly Trend Data Points: {trend_len}, Trajectory: {r9.json().get('trend_direction', 'calculated')}",
    )
)

# 10. GET /api/v1/analytics/cohort
r10 = client.get("/api/v1/analytics/cohort", headers=headers)
results.append(
    (
        "GET",
        "/api/v1/analytics/cohort",
        r10.status_code,
        r10.status_code == 200,
        f"Total Cohort Size: {r10.json().get('total_students')}, Risk Distribution: {r10.json().get('risk_distribution')}",
    )
)

# 11. GET /api/v1/analytics/resources
r11 = client.get("/api/v1/analytics/resources", headers=headers)
results.append(
    (
        "GET",
        "/api/v1/analytics/resources",
        r11.status_code,
        r11.status_code == 200,
        f"Total Catalog Resources: {r11.json().get('total_resources')}, Top Resources Tracked: {len(r11.json().get('top_resources', []))}",
    )
)

# 12. GET /api/v1/models/status
r12 = client.get("/api/v1/models/status", headers=headers)
champ = r12.json().get("champion_risk_model", {})
results.append(
    (
        "GET",
        "/api/v1/models/status",
        r12.status_code,
        r12.status_code == 200,
        f"Champion: {champ.get('name')}, F1 Score: {champ.get('f1_score')}, Status: {champ.get('status')}",
    )
)

print("\n" + "=" * 95)
print(f"{'#':<3} {'METHOD':<6} {'ENDPOINT PATH':<36} {'STATUS':<8} {'VERIFIED':<10} {'DETAILS'}")
print("=" * 95)
for idx, (method, path, status, ok, detail) in enumerate(results, 1):
    ver_str = "PASS" if ok else "FAIL"
    print(f"{idx:<3} {method:<6} {path:<36} {status:<8} {ver_str:<10} {detail}")
print("=" * 95 + "\n")
