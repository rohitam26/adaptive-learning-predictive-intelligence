# REST API Specification
## Adaptive Learning & Predictive Intelligence Platform

### 1. Base URL & Authentication
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication Scheme**: OAuth2 Password Bearer with JWT Token (`Authorization: Bearer <TOKEN>`)

### 2. Authentication Endpoints

#### `POST /auth/register`
Creates a new user profile and returns access token.
```json
// Request Body
{
  "email": "student2@mintx.edu",
  "password": "Pass123!",
  "full_name": "Alex Mercer",
  "role": "student",
  "student_id": 100002
}
```

#### `POST /auth/login`
Authenticates existing credentials and returns JWT bearer token.
```json
// Request Body
{
  "email": "student@mintx.edu",
  "password": "Pass123!"
}
// Response 200 OK
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "role": "student",
  "student_id": 100001,
  "full_name": "Demo Student"
}
```

### 3. Student Intelligence Endpoints

#### `GET /students/me`
Fetches currently authenticated user metadata.

#### `GET /students/{student_id}/profile`
Returns demographic records, module enrollments, and prior attempt counts.

#### `GET /students/{student_id}/performance`
Returns formal assessment completion history, score statistics, and submission delay audit.

#### `GET /students/{student_id}/risk`
Returns model risk probability, assigned risk tier, confidence metric, and SHAP factor contributions.

#### `GET /students/{student_id}/recommendations`
Returns top-K personalized resources with graph affinity scores and pedagogical rationale.

#### `GET /students/{student_id}/engagement`
Returns VLE click volume, unique resources, daily velocity, and 12-week temporal trend.

### 4. Cohort & System Analytics

#### `GET /analytics/cohort`
Cohort-wide at-risk percentage, risk tier distribution, and global SHAP feature importance rankings.

#### `GET /analytics/resources`
Inventory of curriculum resources by activity category and top-engaged course assets.

#### `GET /models/status`
Operational status of champion supervised models, PyTorch deep learning models, and bipartite graph metrics.

#### `GET /health`
System operational status, database connectivity, and model artifact availability.
