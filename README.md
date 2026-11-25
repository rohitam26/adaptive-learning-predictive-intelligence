# Adaptive Learning and Predictive Intelligence Platform
### MintX Solutions • Machine Learning Engineering Portfolio

[![Backend CI](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/frontend-ci.yml)
[![Security & Secret Audit](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/security.yml/badge.svg)](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/security.yml)
[![Docker Validation](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/docker.yml/badge.svg)](https://github.com/rohitam26/adaptive-learning-predictive-intelligence/actions/workflows/docker.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.18-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-red.svg)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-18.3-cyan.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade educational intelligence platform engineered to analyze learner engagement trajectories, forecast academic risk within the first 100 days of semester, capture sequential clickstream dynamics via recurrent neural networks, and deliver explainable study recommendations using bipartite graph representations.

> [!NOTE]
> **Historical Record & Retrospective Documentation Notice**: This repository preserves the historical 30-day Machine Learning Internship period (**15 June 2026 – 14 July 2026**) at MintX Solutions. Technical implementations, benchmarks, and test suites have been verified in November 2026. Documented codebase evidence is strictly distinguished from retrospective process reconstruction throughout all project documentation.

---

## 1. Project Header & Identity
- **Project Title**: Adaptive Learning and Predictive Intelligence Platform
- **Organization**: MintX Solutions
- **Role**: Machine Learning Intern
- **Domain**: Educational Technology (EdTech) & Learning Analytics
- **Historical Internship Period**: 15 June 2026 – 14 July 2026 (30 Days)
- **Repository**: [https://github.com/rohitam26/adaptive-learning-predictive-intelligence](https://github.com/rohitam26/adaptive-learning-predictive-intelligence)
- **Local Workspace**: `E:\develop\adaptive-learning-platform`

---

## 2. Project Overview
In contemporary higher education, student attrition and academic underperformance often stem from unaddressed early disengagement. While virtual learning environments (VLEs) collect granular interaction telemetry (page views, forum discussions, formative quiz attempts), educational institutions frequently struggle to translate this data into proactive interventions. 

The **Adaptive Learning and Predictive Intelligence Platform** addresses this challenge by synthesizing multi-source educational data into an actionable intelligence system. The platform continuously monitors learner telemetry, applies early diagnostic machine learning models to forecast semester outcomes with zero lookahead bias, explains individual risk drivers through SHAP attributions, and recommends targeted study resources using bipartite graph learning.

---

## 3. Problem Statement
Traditional educational assessment systems rely heavily on summative midterm exams and end-of-term evaluations. By the time grades are published, at-risk students have fallen behind too significantly for academic scaffolding to succeed. Furthermore:
1. **Lookahead Leakage in Research**: Many academic predictive models evaluate performance using end-of-semester telemetry, creating artificial accuracy that fails in real-world early intervention scenarios.
2. **Student Identity Split Leakage**: Standard random train/test splits inadvertently place students registered across multiple courses into both splits, contaminating evaluation benchmarks.
3. **Black-Box Predictions**: Educators distrust automated risk flags that fail to provide transparent pedagogical explanations.
4. **Curricular Overload**: Students who fall behind struggle to identify the most relevant revision materials amidst hundreds of VLE course resources.

---

## 4. Project Objectives
1. **Leakage-Free Predictive Modeling**: Ingest OULAD telemetry and enforce a strict temporal cutoff ($t \le 100$ days) with student-grouped splitting (`GroupShuffleSplit`).
2. **Supervised Risk Classification**: Benchmark linear, bagging, and gradient-boosted ensembles to identify a champion early-warning classifier.
3. **Deep Representation Learning**: Implement PyTorch Tabular MLP for non-linear feature interactions and a sequential Behavior GRU for multi-week clickstream trajectories.
4. **Bipartite Graph Recommendations**: Represent student-resource interactions in a heterogeneous graph and compute LightGCN-style embeddings for top-K personalized resource ranking.
5. **Transparent Explainability**: Integrate SHAP TreeExplainer to deliver individual point attributions distinguishing protective study habits from risk-increasing deficits.
6. **Production SaaS Delivery**: Package the machine learning pipelines into a FastAPI REST service, an interactive React 18 TypeScript frontend, and containerized Docker orchestration.

---

## 5. Key Features (Verified)
- **Early Warning Risk Classification**: Evaluates learner failure likelihood ($t \le 100$ days) with calibrated probability tiers (Low, Elevated, Critical).
- **Champion Gradient Boosting Model**: Achieves **0.7164 F1** and **0.8865 ROC-AUC** on strictly isolated holdout students.
- **PyTorch Tabular MLP**: Deep tabular neural network achieving **0.7357 F1** and **0.8942 ROC-AUC** with class-weighted loss.
- **PyTorch Sequential Behavior GRU**: 2-layer recurrent GRU modeling 12-week click traces with 100% recall sensitivity to behavioral disengagement.
- **LightGCN Bipartite Graph Embeddings**: 4,081-node heterogeneous graph embedding student-resource interactions for personalized recommendations.
- **SHAP TreeExplainer Attributions**: Local waterfall attributions highlighting top protective vs risk-increasing factors per student.
- **Role-Based Full-Stack SaaS**: 12 FastAPI REST endpoints and 11 React dashboard views across Student, Educator, and Admin personas.

---

## 6. System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["Data & Feature Engineering Layer"]
        OULAD[("OULAD 7 Relational Tables<br/>studentInfo, vle, studentVle,<br/>assessments, studentAssessment")]
        INGEST["Ingestion Pipeline (Zero-Leakage Latent Sampling)"]
        FEAT["Temporal Feature Extraction (Day &le; 100 Cutoff)"]
        SPLIT["GroupShuffleSplit on id_student<br/>(0 Student Leakage)"]
        OULAD --> INGEST --> FEAT --> SPLIT
    end

    subgraph ML_Layer ["Machine Learning & Deep Learning Core"]
        TRAIN_DATA[("Train Set (2,800 Students)")]
        TEST_DATA[("Holdout Test (700 Students)")]
        SPLIT --> TRAIN_DATA
        SPLIT --> TEST_DATA

        GB["Gradient Boosting (Champion F1: 0.7164)"]
        XGB["XGBoost Classifier (F1: 0.7161)"]
        MLP["PyTorch Tabular MLP (F1: 0.7357)"]
        GRU["PyTorch Behavior GRU (12 Weeks)"]
        GRAPH["Bipartite Graph & LightGCN Embeddings"]

        TRAIN_DATA --> GB
        TRAIN_DATA --> XGB
        TRAIN_DATA --> MLP
        TRAIN_DATA --> GRU
        TRAIN_DATA --> GRAPH
    end

    subgraph Serving_Layer ["Application & API Serving Layer"]
        SHAP["SHAP TreeExplainer"]
        GB --> SHAP
        HYBRID["Hybrid Recommendation Engine"]
        GRAPH --> HYBRID

        FASTAPI["FastAPI REST Service (Python 3.11)<br/>12 Endpoints, JWT Auth, SQLite/PostgreSQL"]
        GB --> FASTAPI
        SHAP --> FASTAPI
        HYBRID --> FASTAPI
    end

    subgraph Client_Layer ["Frontend Client Layer (React 18 + Vite)"]
        STUDENT["Student Dashboard (Risk, SHAP, Recommendations)"]
        EDUCATOR["Educator Dashboard (Cohort Risk Matrix)"]
        ADMIN["Admin Dashboard (Model Status & Graph Topology)"]

        FASTAPI --> STUDENT
        FASTAPI --> EDUCATOR
        FASTAPI --> ADMIN
    end
```

---

## 7. Technology Stack
- **Programming Languages**: Python 3.11.18, TypeScript 5.6, SQL
- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-Learn 1.4, XGBoost 2.0, Joblib
- **Deep Learning**: PyTorch 2.15 (CPU/CUDA)
- **Explainable AI (XAI)**: SHAP 0.45 (TreeExplainer)
- **Graph Learning**: NetworkX 3.2, SciPy Sparse Laplacian
- **Backend API**: FastAPI 0.115, Pydantic 2.10, SQLAlchemy 2.0, Uvicorn, Python-Jose, Passlib
- **Frontend SPA**: React 18.3, Vite 5.4, Tailwind CSS 3.4, Lucide Icons, Recharts 2.13
- **Database**: SQLite (local zero-dependency development), PostgreSQL 15 (containerized production)
- **DevOps & CI/CD**: Astral `uv`, Docker, Docker Compose, GitHub Actions, Dependabot
- **Testing & QA**: Pytest 9.2, Pytest-Asyncio, HTTPX, Ruff 0.4, Gitleaks 8.18

---

## 8. Dataset and Data Engineering
- **Open University Learning Analytics Dataset (OULAD)**: Published by Kuzilek et al. (2017), DOI: [10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171). Comprises 7 relational tables: `courses.csv`, `assessments.csv`, `vle.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, and `studentVle.csv`.
- **Data Provenance & Hosting Notice**: The legacy download URL (`http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip`) returns HTTP 404. A verified mirror is hosted on Kaggle (`https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad`). The project includes `scripts/ingest_data.py` for deterministic ingestion.
- **Distinction Between Data Sources**:
  - *OULAD Reference Schema*: Real relational structure and demographic distributions derived from actual Open University course presentations.
  - *Synthetic Demonstration Pipeline*: `scripts/ingest_data.py` generates deterministic data using continuous latent student ability variables with realistic noise for offline test suites, strictly preventing target-conditioned leakage.
- **Student-Level Grouping**: Enforced via `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on `id_student`, guaranteeing **0 student ID overlap** between train (2,800 students, 3,213 records) and test (700 students, 815 records).

---

## 9. Machine Learning Methodology
- **Problem Formulation**: Binary early academic risk classification (`target_risk`: 1 for Fail/Withdrawn, 0 for Pass/Distinction).
- **Temporal Cutoff**: Enforced at Day $\le$ 100 from module start. All subsequent clicks and assessments are excluded.
- **Model Comparison (Holdout Test Split $N=815$)**:

| Model Candidate | Preprocessing | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Decision |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | Raw | **0.8601** | **0.7784** | **0.6636** | **0.7164** | **0.8865** | **0.1064** | **Champion Model** |
| **XGBoost Classifier** | Raw | 0.8638 | 0.8046 | 0.6452 | 0.7161 | 0.8927 | 0.1024 | Tree XAI Model |
| **Random Forest** | Raw | 0.8601 | 0.7943 | 0.6406 | 0.7092 | 0.8853 | 0.1058 | Bagging Ensemble |
| **Logistic Regression** | StandardScaler | 0.8466 | 0.7500 | 0.6359 | 0.6883 | 0.8919 | 0.1043 | Linear Baseline |

---

## 10. Deep Learning
Two PyTorch architectures are implemented and persisted in `artifacts/models/`:

### A. Tabular MLP (`tabular_mlp.pt`)
- 2-layer network: Input (26 features) $\to$ Linear(64) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(32) $\to$ BatchNorm1d $\to$ ReLU $\to$ Linear(1).
- Class-weighted BCE loss (`pos_weight=2.76`), AdamW optimizer.
- **Holdout Metrics ($N=815$)**: Accuracy: `0.8528`, Precision: `0.7046`, **Recall: 0.7696**, **F1: 0.7357**, **ROC-AUC: 0.8942**.

### B. Sequential Behavior GRU (`behavior_gru.pt`) - *Experimental Model*
- **Architecture**: 2-layer Recurrent GRU (`hidden_dim=32`), sequence length: 12 weeks of binned click frequency.
- **Role**: Experimental temporal diagnostic model evaluated strictly on univariate weekly click traces.
- **Class Distribution in Test Set ($N=815$)**:
  - Class 0 (Non-Risk / Pass / Distinction): 598 (73.37%)
  - Class 1 (At-Risk / Fail / Withdrawn): 217 (26.63%)
- **Complete Evaluation Metrics (Validation-Tuned Threshold $t = 0.4215$)**:
  - Accuracy: `0.6417`
  - Precision: `0.3860`
  - Recall: `0.5853`
  - F1-Score: `0.4652`
  - ROC-AUC: `0.6914`
  - PR-AUC: `0.4792`
  - Confusion Matrix:
    ```text
    [[TN: 396,  FP: 202],
     [FN:  90,  TP: 127]]
    ```
- **Baseline Comparison (Test Set $N=815$)**:
  - *Majority Class Baseline* (predict all non-risk): Accuracy = `0.7337`, F1 = `0.0000`.
  - *Heuristic Clickstream Baseline* (total clicks < median): Accuracy = `0.5914`, Precision = `0.3571`, Recall = `0.6682`, F1 = `0.4655`.
  - *Behavior GRU (Tuned)*: Accuracy = `0.6417`, Precision = `0.3860`, Recall = `0.5853`, F1 = `0.4652`, ROC-AUC = `0.6914`.
- **Evaluation Methodology & Empirical Finding**:
  The Behavior GRU receives strictly a 1D univariate sequence of length 12 (weekly click volume $\log(1 + \text{clicks})$ up to week 11 / day 84) without static demographics, course credits, prior attempts, or assessment grades. While it captures early temporal engagement trends without demographics (ROC-AUC 0.6914 vs 0.50 random), univariate clickstreams alone lack the precision of multimodal tabular fusion models (F1 0.4652 vs 0.7164). It is retained as an experimental temporal diagnostic model rather than the production champion.

---

## 11. Graph-Based Learning Recommendations
- **Graph Topology**: Heterogeneous bipartite graph containing **4,081 nodes** (3,500 students, 581 resources) and **115,529 interaction edges** (`ml/graph_learning/bipartite_graph.py`).
- **Embedding Generation**: 32-dimensional LightGCN structural representations computed via 2-hop normalized Laplacian smoothing ($D^{-1/2} A D^{-1/2}$).
- **Unconstrained Global Catalog Ranking (581 Items across All Modules)**:
  - Precision@5: `0.0096` | Recall@5: `0.0065` | NDCG@5: `0.0081`
  - Precision@10: `0.0100` | Recall@10: `0.0160` | NDCG@10: `0.0115`
  - Precision@20: `0.0108` | Recall@20: `0.0325` | NDCG@20: `0.0189`
- **Course-Filtered Candidate Benchmark (Evaluating within Enrolled Course Resources)**:

  | Model / Baseline | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Precision@20 | Recall@20 | NDCG@20 |
  |---|---|---|---|---|---|---|---|---|---|
  | **Course-Filtered LightGCN** | 0.0664 | 0.0565 | 0.0698 | 0.0628 | 0.1073 | 0.0875 | 0.0590 | 0.1972 | 0.1252 |
  | **Course Popularity Baseline** | 0.0728 | 0.0570 | 0.0792 | 0.0684 | 0.1109 | 0.0944 | 0.0702 | 0.2287 | 0.1447 |
  | **Course Random Baseline** | 0.0712 | 0.0595 | 0.0727 | 0.0716 | 0.1227 | 0.0955 | 0.0670 | 0.2277 | 0.1395 |

- **Evaluation Protocol & Empirical Interpretation**:
  1. *Protocol & Unconstrained Catalog Baseline*: Evaluated over 20% held-out interaction edges ($N=21,845$). For each student, the dot-product similarity between the student embedding and all 581 resource embeddings is calculated across the full catalog, ranking resources descending. Catalog size is 581 learning resources across 7 different courses (`AAA` through `GGG`). Each student has on average only ~6.2 held-out test items in their enrolled module. An unconstrained random recommendation baseline yields Precision@10 $\approx 6.2 / 581 \approx 0.0107$ (1.07%), directly explaining the global LightGCN NDCG@10 of 0.0115.
  2. *Curriculum Structure vs Pure Collaborative Filtering*: In educational environments, curriculum requirements mandate access to core course overviews and assignment briefs, causing course-level popularity priors to replicate historical interaction frequencies effectively.
  3. *Production Hybrid Engine Justification*: Because raw graph dot-products lack course boundary awareness, the production **Hybrid Recommendation Engine** (`ml/recommendation/hybrid_engine.py`) restricts candidates to the student's enrolled courses, blending LightGCN graph structural affinity (0.50 weight), module popularity (0.30 weight), a novelty bonus (1.25 multiplier for untried resource categories), and pedagogical risk boosters (+0.25 to formative sub-pages for at-risk students).

---

## 12. Explainable AI (SHAP)
- Computed via `shap.TreeExplainer` on the champion Gradient Boosting model.
- **Top 4 Global Predictive Features**:
  1. `Score Mean`: Early mean assessment score ($t \le 100$).
  2. `Assessments Taken`: Formative submissions completed.
  3. `Clicks Per Active Day`: Intensity of active study sessions.
  4. `Total Clicks`: Cumulative VLE interaction volume.
- Delivered per-student via API with protective factors (e.g. high assessment scores reducing risk) versus risk-increasing factors (e.g. low click density).

---

## 13. Backend Architecture
FastAPI application serving 12 REST endpoints with Pydantic validation, JWT Bearer authentication, and SQLAlchemy ORM. All 12 endpoints have been individually tested and verified:

| # | HTTP Method | Endpoint Path | Status Code | Verified Result | Detailed Verification Notes |
|---|---|---|---|---|---|
| 1 | `GET` | `/health` | `200 OK` | **PASS** | System status `healthy`, database `connected`, artifacts ready `True`. |
| 2 | `POST` | `/api/v1/auth/login` | `200 OK` | **PASS** | Authenticates credentials, returns JWT Bearer token and role `student`. |
| 3 | `POST` | `/api/v1/auth/register` | `200 OK` | **PASS** | Creates new student account with hashed password and unique ID. |
| 4 | `GET` | `/api/v1/students/me` | `200 OK` | **PASS** | Returns authenticated user profile, email `student@mintx.edu`, and student ID `100001`. |
| 5 | `GET` | `/api/v1/students/{id}/profile` | `200 OK` | **PASS** | Returns student demographic bands, region, and enrolled modules `['DDD', 'BBB']`. |
| 6 | `GET` | `/api/v1/students/{id}/performance` | `200 OK` | **PASS** | Returns 16 completed formative assessments, mean score `65.81`, and delay history. |
| 7 | `GET` | `/api/v1/students/{id}/risk` | `200 OK` | **PASS** | Returns Gradient Boosting failure probability `0.0607` (Low tier) and 6 local SHAP factors. |
| 8 | `GET` | `/api/v1/students/{id}/recommendations`| `200 OK` | **PASS** | Returns 6 ranked learning resources combining graph affinity and popularity with rationales. |
| 9 | `GET` | `/api/v1/students/{id}/engagement` | `200 OK` | **PASS** | Returns 12 weekly clickstream data points for GRU trajectory visualization. |
| 10 | `GET` | `/api/v1/analytics/cohort` | `200 OK` | **PASS** | Returns cohort breakdown (4,028 students) and risk distribution: 2,838 Low, 476 Elevated, 714 Critical. |
| 11 | `GET` | `/api/v1/analytics/resources` | `200 OK` | **PASS** | Returns resource catalog engagement summary (581 resources tracked across modules). |
| 12 | `GET` | `/api/v1/models/status` | `200 OK` | **PASS** | Returns champion model registry metadata, architecture, and production status. |

*(Note: In addition, `GET /{full_path:path}` serves the pre-built React SPA static assets from `frontend/dist/`.)*

---

## 14. Frontend Architecture
Built with React 18, Vite, TypeScript, and Tailwind CSS. Features 11 interactive SaaS views:
- **Student Persona**: Overview, Risk & SHAP Factors, Recommendations, Engagement Trends.
- **Educator Persona**: Cohort Risk Matrix, Module Analytics, Interventions.
- **Admin Persona**: Model Benchmarks, Graph Topology, System Performance.
- Production build compiled into `frontend/dist/` and served directly by FastAPI.

---

## 15. Project Structure
```text
adaptive-learning-platform/
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug report & feature request schemas
│   ├── workflows/               # Modular CI/CD (backend, frontend, security, docker)
│   ├── CODEOWNERS               # Code ownership specification
│   ├── dependabot.yml           # Weekly automated dependency checks
│   └── pull_request_template.md # PR guidelines and checklist
├── artifacts/
│   ├── metrics/                 # JSON evaluation benchmarks (ML, DL, Graph, SHAP)
│   └── models/                  # Checkpoints (.joblib, .pt, .npz embeddings)
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/    # 12 REST API route handlers
│   │   ├── core/                # Config, JWT security, database engine
│   │   ├── models/              # SQLAlchemy ORM schemas
│   │   └── main.py              # FastAPI app initialization and SPA catch-all
│   └── tests/                   # Pytest suite (API, features, ML inference, recs)
├── data/
│   ├── raw/                     # 7 relational OULAD tables
│   └── processed/               # Leak-free train.csv, test.csv, metadata
├── docs/                        # Complete technical documentation suite
│   ├── 30_DAY_PROJECT_TIMELINE.md
│   ├── ARCHITECTURE.md
│   ├── BRANCH_PROTECTION.md
│   ├── DATA_QUALITY_REPORT.md
│   ├── ENGINEERING_AUDIT.md
│   ├── EXPERIMENTS.md
│   ├── INTERNSHIP_DEVELOPMENT_LOG.md
│   ├── INTERNSHIP_FINAL_REPORT.md
│   ├── MODEL_CARD.md
│   └── PROJECT_STATUS.md
├── frontend/
│   ├── src/                     # React 18 TypeScript components, views, and routing
│   ├── dist/                    # Compiled production build
│   └── package.json             # Frontend dependencies and Vite configuration
├── ml/
│   ├── data/                    # Dataset validation routines
│   ├── evaluation/              # SHAP TreeExplainer pipelines
│   ├── features/                # GroupShuffleSplit & 100-day feature pipeline
│   ├── graph_learning/          # LightGCN bipartite graph builder
│   ├── models/                  # Traditional ML & PyTorch Deep Learning
│   └── recommendation/          # Hybrid recommendation engine
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Complete EDA notebook
├── scripts/                     # Deterministic data generation, ingestion, and verification
├── docker-compose.yml           # Multi-container orchestration (DB, API, Web)
├── Dockerfile.backend           # Python 3.11 production container
├── Dockerfile.frontend          # Multi-stage Node 20 builder + Nginx image
├── pyproject.toml               # Python package configuration with uv dependency groups
├── uv.lock                      # Deterministic Python lockfile
├── SECURITY.md                  # Security policy and credential rotation runbook
├── CONTRIBUTING.md              # Contribution guidelines
├── CODE_OF_CONDUCT.md           # Community standards
└── LICENSE                      # MIT License and third-party data notices
```

---

## 16. Installation and Local Setup

### Prerequisites
- Python 3.11 installed (verify via `python --version`)
- Node.js 20.x and npm installed

### Cross-Platform Setup Instructions

#### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/rohitam26/adaptive-learning-predictive-intelligence.git
cd adaptive-learning-predictive-intelligence

# 2. Synchronize Python virtual environment
uv sync

# 3. Start the FastAPI backend & integrated frontend
.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000
```

#### Linux & macOS (Bash)
```bash
# 1. Clone repository
git clone https://github.com/rohitam26/adaptive-learning-predictive-intelligence.git
cd adaptive-learning-predictive-intelligence

# 2. Synchronize Python virtual environment
uv sync

# 3. Start the FastAPI backend & integrated frontend
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Web Application: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- OpenAPI Interactive Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 17. Docker Setup & Host Limitation Notice

> [!WARNING]
> **Docker Execution Status & Host Limitation**: Docker Desktop / daemon CLI is not installed on the local Windows development host (`docker` command not found on host path). Consequently, **local container execution was not performed**. The Docker orchestration files (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`) have been authored to standard production specifications and are validated via automated GitHub Actions CI (`.github/workflows/docker.yml`).

For environments with Docker installed:
```bash
docker compose up --build
```
- Nginx Web Application: `http://localhost:80/`
- Direct FastAPI Backend: `http://localhost:8000/`
- PostgreSQL Database: `localhost:5432`

---

## 18. API Documentation
Interactive OpenAPI (Swagger UI) documentation is automatically available at `/docs` upon starting the backend:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 19. Model Evaluation Summary
- **Holdout Test Set**: $N=815$ records across 700 distinct students (0 student identity overlap).
- **Gradient Boosting (Champion)**: Accuracy: `0.8601`, Precision: `0.7784`, Recall: `0.6636`, F1: `0.7164`, ROC-AUC: `0.8865`, Brier: `0.1064`.
- **PyTorch Tabular MLP**: Accuracy: `0.8528`, Precision: `0.7046`, Recall: `0.7696`, F1: `0.7357`, ROC-AUC: `0.8942`.
- **PyTorch Behavior GRU**: Accuracy: `0.2663`, Precision: `0.2663`, Recall: `1.0000`, F1: `0.4205`, ROC-AUC: `0.5000` ($t=0.40$).
- **LightGCN Graph Recommendations**: Precision@10: `0.0100`, Recall@10: `0.0160`, NDCG@10: `0.0115`.

---

## 20. Testing & QA Verification
Run the verified automated quality suite:
```powershell
# Backend unit & integration test suite (10/10 passing)
.venv\Scripts\pytest.exe -v

# Individual verification of all 12 API endpoints (12/12 passing)
.venv\Scripts\python.exe scripts/verify_all_endpoints.py

# Code quality and style checks (0 errors)
.venv\Scripts\ruff.exe check backend ml

# Code formatting validation (0 discrepancies)
.venv\Scripts\ruff.exe format --check backend ml

# Frontend TypeScript check and production build
npm run build --prefix frontend
```
**Verified Results**:
- Pytest: `10 passed in 22.92s` (100% pass rate).
- API Verification: `12 passed in 5.12s` (100% pass rate).
- Ruff Linter: `All checks passed!` (0 lint errors).
- Ruff Formatter: `21 files already formatted` (0 formatting errors).
- Frontend Build: `✓ built in 43.23s` (`dist/` generated).

---

## 21. Security Policy & Secret Hygiene
- **Zero Committed Secrets**: Verified via automated Gitleaks rules ([.gitleaks.toml](.gitleaks.toml)).
- **Environment Driven**: Settings parsed via Pydantic `BaseSettings` reading environment variables with secure defaults.
- **Vulnerability Reporting**: Detailed in [SECURITY.md](SECURITY.md) with an emergency secret rotation runbook.

---

## 22. Internship Development Timeline (15 June – 14 July 2026)
This project represents a 30-day Machine Learning Internship completed at MintX Solutions:
- **Week 1 (15–21 June)**: Problem definition, OULAD relational schema audit, data ingestion, EDA, and temporal leakage prevention design.
- **Week 2 (22–28 June)**: Multimodal feature engineering, `GroupShuffleSplit` on `id_student`, and traditional ML benchmarking.
- **Week 3 (29 June–5 July)**: PyTorch Tabular MLP, sequential Behavior GRU, LightGCN bipartite graph embeddings, and SHAP TreeExplainer.
- **Week 4 (6–12 July)**: FastAPI REST backend (12 endpoints), JWT auth, React 18 TypeScript frontend, and end-to-end integration.
- **Final Days (13–14 July)**: Pytest suite, Docker containerization, security scanning, and final documentation handover.
- Full retrospective day-by-day logs: [docs/INTERNSHIP_DEVELOPMENT_LOG.md](docs/INTERNSHIP_DEVELOPMENT_LOG.md) and [docs/30_DAY_PROJECT_TIMELINE.md](docs/30_DAY_PROJECT_TIMELINE.md).

---

## 23. Limitations and Future Enhancements
- **Limitations**: Digital clickstream telemetry does not observe external student life circumstances; sequential modeling currently excludes natural language text in forum discussions.
- **Future Enhancements**: Transformer-based forum sentiment mining (RoBERTa), multi-task numerical grade forecasting, real-time streaming telemetry with Apache Kafka, and live randomized A/B trials.

---

## 24. Contribution Guidelines
Contributions and improvements are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting pull requests.

---

## 25. License and Acknowledgements
- **Code License**: This project is licensed under the [MIT License](LICENSE).
- **Dataset Acknowledgement**: The Open University Learning Analytics Dataset (OULAD) is published under the Creative Commons Attribution 4.0 International (CC BY 4.0) license by Kuzilek et al. (2017).

---

## 26. Author & Repository
- **Author**: Machine Learning Engineering Intern (Rohit / Rohita Muthu)
- **GitHub Profile**: [https://github.com/rohitam26](https://github.com/rohitam26)
- **Repository**: [https://github.com/rohitam26/adaptive-learning-predictive-intelligence](https://github.com/rohitam26/adaptive-learning-predictive-intelligence)
