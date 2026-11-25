# Requirements Specification
## Adaptive Learning & Predictive Intelligence Platform

### 1. Functional Requirements

#### 1.1 Learner Analytics & Early Risk Classification
- **FR-01**: Ingest student demographics, registration timestamps, assessment scores, and VLE interaction telemetry.
- **FR-02**: Calculate behavioral engagement metrics (total clicks, active study days, clicks per active day, activity type concentration).
- **FR-03**: Train and compare multiple supervised classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
- **FR-04**: Classify students into risk tiers (Low, Elevated, Critical) using early temporal windows (<= day 100) to prevent future data leakage.

#### 1.2 Deep Sequential Modeling
- **FR-05**: Implement PyTorch Tabular MLP for non-linear feature interaction modeling.
- **FR-06**: Implement PyTorch Behavior GRU to process 12 weekly interaction click sequences and forecast risk progression.

#### 1.3 Graph Learning & Recommendations
- **FR-07**: Construct a student-resource bipartite graph containing student nodes, resource nodes, and interaction edges.
- **FR-08**: Compute LightGCN multi-layer neighborhood aggregation embeddings.
- **FR-09**: Evaluate top-K recommendation rankings using Precision@K, Recall@K, and NDCG@K.
- **FR-10**: Provide a hybrid personalization engine combining graph affinity, popularity, and academic risk context with human-interpretable rationale.

#### 1.4 Explainable AI (SHAP)
- **FR-11**: Generate cohort-wide global feature importance rankings.
- **FR-12**: Produce individual student prediction explanations with feature attribution scores and directional impact.

#### 1.5 Web Application & Dashboards
- **FR-13**: Implement JWT token authentication supporting Student, Educator, and Administrator roles.
- **FR-14**: Expose all required REST endpoints matching the API contract.
- **FR-15**: Provide responsive dashboards for student self-monitoring, educator cohort management, and administrator model status.

### 2. Non-Functional Requirements
- **NFR-01 Performance**: API latency below 150ms for profile, performance, and recommendation endpoints.
- **NFR-02 Reliability**: Automated test coverage exceeding 80% across critical API flows.
- **NFR-03 Reproducibility**: Seeded data generation, deterministic feature splits, and pinned dependencies via `pyproject.toml`.
- **NFR-04 Portability**: Containerized deployment via Docker and Docker Compose supporting SQLite and PostgreSQL backends.
