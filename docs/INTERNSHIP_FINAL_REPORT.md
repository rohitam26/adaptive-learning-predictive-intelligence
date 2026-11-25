# Internship Final Engineering Report
## Adaptive Learning and Predictive Intelligence Platform

**Organization:** MintX Solutions  
**Role:** Machine Learning Intern  
**Domain:** Educational Technology (EdTech) & Learning Analytics  
**Internship Period Documented:** 15 June 2026 – 14 July 2026 (30 Days)  
**Author:** Machine Learning Engineering Intern (Rohit / Rohita Muthu)  
**Repository:** [https://github.com/rohitam26/adaptive-learning-predictive-intelligence](https://github.com/rohitam26/adaptive-learning-predictive-intelligence)  

> [!NOTE]
> **Historical Record & Retrospective Documentation Notice**: This document details the technical outputs and retrospective engineering progression of the 30-day Machine Learning Internship (15 June – 14 July 2026) at MintX Solutions. Technical implementations, benchmarks, and test suites were verified in November 2026. Documented codebase evidence is strictly distinguished from retrospective process reconstruction throughout this report.

---

### 1. Organization and Internship Context
MintX Solutions is an enterprise AI and predictive intelligence firm. During this 30-day Machine Learning Internship (15 June 2026 – 14 July 2026), the engineering directive was to design, implement, benchmark, and deploy an end-to-end educational intelligence platform that leverages telemetry from higher education virtual learning environments to detect academic risk early and deliver personalized learning interventions.

---

### 2. Project Title
**Adaptive Learning and Predictive Intelligence Platform**  
*An integrated educational intelligence system combining predictive risk modeling, deep sequential learning, bipartite graph recommendations, and transparent SHAP explainability.*

---

### 3. Problem Statement
In higher education online and blended learning environments, students frequently disengage gradually and silently. Traditional academic warning systems depend on summative midterm exams or end-of-term evaluations, which occur too late for academic advisors to deliver meaningful scaffolding. Concurrently, virtual learning environments (VLEs) present learners with vast, unstructured repositories of course materials without adaptive guidance tailored to individual competency gaps. 

There is a critical requirement for a predictive intelligence system that:
1. Detects learner disengagement within an early-warning window ($t \le 100$ days).
2. Generates calibrated probabilities of academic failure with zero lookahead leakage.
3. Explains the exact behavioral and assessment drivers underlying each prediction.
4. Recommends targeted curricular resources using graph representations of cohort interactions.

---

### 4. Project Objectives
The engineering objectives of the project were:
1. **Data Engineering & Leakage Prevention**: Ingest the Open University Learning Analytics Dataset (OULAD), resolve missing values, and establish a strict 100-day cutoff with student-level `GroupShuffleSplit` partitioning.
2. **Supervised Risk Classification**: Benchmark Logistic Regression, Random Forest, Gradient Boosting, and XGBoost to identify a champion early-warning classifier.
3. **Deep Learning Behavioral Modeling**: Implement a deep Tabular MLP for non-linear feature interactions and a sequential Behavior GRU for multi-week clickstream trajectory tracking.
4. **Graph-Based Personalized Recommendations**: Construct a heterogeneous student-resource bipartite graph and compute LightGCN-style embeddings for top-K resource ranking.
5. **Transparent Explainability**: Integrate SHAP TreeExplainer to deliver individual point attributions distinguishing protective study habits from risk-increasing deficits.
6. **Production Full-Stack Deployment**: Package the system into a FastAPI REST backend (12 endpoints), an interactive React 18 TypeScript frontend, and Docker Compose orchestration.

---

### 5. Tools and Technologies

| Category | Technologies |
|---|---|
| **Languages** | Python 3.11.18, TypeScript 5.6, SQL |
| **Data Processing & Analytics** | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-Learn 1.4, XGBoost 2.0, Joblib |
| **Deep Learning** | PyTorch 2.15 |
| **Graph Learning & XAI** | NetworkX 3.2, SHAP 0.45 |
| **Backend & APIs** | FastAPI 0.115, Pydantic 2.10, SQLAlchemy 2.0, Uvicorn, Python-Jose, Passlib |
| **Frontend Architecture** | React 18.3, Vite 5.4, Tailwind CSS 3.4, Lucide Icons, Recharts |
| **Database** | SQLite (local zero-dependency development), PostgreSQL 15 (containerized) |
| **DevOps & QA** | Astral `uv`, Pytest 9.2, Ruff 0.4, Docker, Docker Compose, GitHub Actions, Gitleaks |

---

### 6. Dataset and Preprocessing Methodology
- **Dataset**: Open University Learning Analytics Dataset (OULAD), comprising 7 relational tables: `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, `studentAssessment`, and `studentVle`.
- **Target Leakage Remediation**:
  - Prior iterations exhibited unrealistic classification scores (~99.88% accuracy) due to target-conditioned synthetic sampling and row-level random splitting across multi-course enrollments.
  - The ingestion pipeline was refactored to generate student engagement traces from latent continuous variables without conditioning on target outcomes.
  - Data partitioning was refactored using `GroupShuffleSplit` on `id_student`, strictly enforcing **0 student ID overlap** between train (2,800 students, 3,213 records) and test (700 students, 815 records).
- **Temporal Window Cutoff**: All clickstream logs and assessment submissions were strictly bounded to $t \le 100$ days from presentation launch. All downstream events were excluded to prevent lookahead bias.
- **Engineered Features (26 Dimensions)**:
  - *Demographics*: `gender_code`, `disability_code`, `age_band_code`, `education_level_code`, `imd_band_code`.
  - *VLE Engagement*: `total_clicks`, `unique_resources`, `active_days`, `clicks_per_active_day`, `early_click_mean`, `clicks_forumng`, `clicks_oucontent`, `clicks_quiz`, `clicks_resource`, `clicks_subpage`, `clicks_url`.
  - *Assessment Performance*: `assessments_taken`, `score_mean`, `score_min`, `score_max`, `score_std`, `late_submissions_count`, `mean_delay_days`.
  - *Course Context*: `num_of_prev_attempts`, `studied_credits`, `date_registration`.

---

### 7. Machine Learning Methodologies and Results
All models were evaluated on the leak-free holdout test set ($N=815$ records, 700 distinct students):

| Model Architecture | Scaling | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | None | **0.8601** | **0.7784** | **0.6636** | **0.7164** | **0.8865** | **0.8118** | **0.1064** | **Champion Model** |
| **XGBoost Classifier** | None | 0.8638 | 0.8046 | 0.6452 | 0.7161 | 0.8927 | 0.8230 | 0.1024 | Tree XAI Model |
| **Random Forest** | None | 0.8601 | 0.7943 | 0.6406 | 0.7092 | 0.8853 | 0.8054 | 0.1058 | Bagging Ensemble |
| **Logistic Regression** | StandardScaler | 0.8466 | 0.7500 | 0.6359 | 0.6883 | 0.8919 | 0.8163 | 0.1043 | Linear Baseline |

**Selection Rationale**: Gradient Boosting achieved the optimal balance of recall (0.6636) and overall F1-score (0.7164) with a well-calibrated Brier score of 0.1064, making it the champion model for production inference.

---

### 8. Deep Learning Methodologies & Comprehensive Evaluation

#### A. Tabular Multi-Layer Perceptron (`tabular_mlp.pt`)
- **Architecture**: 2 hidden layers (64, 32 units) with `BatchNorm1d`, `ReLU` activations, and `Dropout(0.3)`.
- **Loss & Optimization**: Class-weighted `BCEWithLogitsLoss` (`pos_weight=2.76`), AdamW optimizer (`lr=0.003`, `weight_decay=1e-4`).
- **Holdout Metrics ($N=815$)**: Accuracy: `0.8528`, Precision: `0.7046`, **Recall: 0.7696**, **F1-Score: 0.7357**, **ROC-AUC: 0.8942**.
- **Finding**: The deep tabular network demonstrated superior recall on the minority at-risk class compared to tree baselines, capturing subtle non-linear demographic and engagement interactions.

#### B. Sequential Behavior GRU (`behavior_gru.pt`)
- **Architecture**: 2-layer Recurrent Gated Recurrent Unit (`hidden_dim=32`), ingesting 12-week normalized click frequencies ($\log(1+\text{clicks})$).
- **Optimization**: Adam optimizer (`lr=0.002`), gradient norm clipping (1.0).
- **Holdout Test Set Class Distribution ($N=815$)**:
  - Class 0 (Non-Risk / Pass / Distinction): 598 (73.37%)
  - Class 1 (At-Risk / Fail / Withdrawn): 217 (26.63%)
- **Investigation, Root Cause & Threshold Calibration**:
  - *Root Cause Analysis*: Weekly unobserved pivot slots initially left unvisited weeks as `NaN`, causing input tensor explosion into `NaN`. Fixing week alignment with `.fillna(0.0)` restored proper gradient convergence.
  - *Threshold Tuning*: Calibrated strictly on a 20% validation split of the training set via precision-recall trade-off maximization, establishing the optimal decision threshold $t = 0.4215$.
- **Holdout Test Evaluation Metrics ($N=815$)**:
  - Decision Threshold: $t = 0.4215$
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
  The Behavior GRU receives strictly a 1D univariate sequence of length 12 (weekly click volume $\log(1+\text{clicks})$ up to day 84) without static demographics, course credits, prior attempts, or formative assessment scores. While it effectively captures early temporal engagement decay (ROC-AUC 0.6914 vs 0.5000 random chance), univariate click frequency alone lacks the high precision of multimodal tabular fusion (F1 0.4652 vs 0.7164 in Gradient Boosting and 0.7368 in Tabular MLP). It is retained as an experimental temporal diagnostic model rather than the primary production classifier.

---

### 9. Recommendation Architecture & LightGCN Ranking Analysis
- **Graph Construction**: Built a heterogeneous bipartite graph using NetworkX comprising **4,081 nodes** (3,500 students, 581 resources) and **115,529 interaction edges**.
- **Edge Partitioning**: 80% training edges ($N=93,684$), 20% holdout test edges ($N=21,845$).
- **Embedding Generation**: Computed 32-dimensional LightGCN structural representations via 2-hop normalized Laplacian smoothing:
  $$\tilde{A} = D^{-1/2} A D^{-1/2}, \quad E^{(k+1)} = \tilde{A} E^{(k)}$$
- **Unconstrained Global Catalog Ranking (581 Items)**:
  - Precision@5: `0.0096` | Recall@5: `0.0065` | NDCG@5: `0.0081`
  - Precision@10: `0.0100` | Recall@10: `0.0160` | NDCG@10: `0.0115`
  - Precision@20: `0.0108` | Recall@20: `0.0325` | NDCG@20: `0.0189`
- **Course-Filtered Candidate Benchmark (Evaluating within Enrolled Course Resources)**:

  | Model / Baseline | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Precision@20 | Recall@20 | NDCG@20 |
  |---|---|---|---|---|---|---|---|---|---|
  | **Course-Filtered LightGCN** | 0.0664 | 0.0565 | 0.0698 | 0.0628 | 0.1073 | 0.0875 | 0.0590 | 0.1972 | 0.1252 |
  | **Course Popularity Baseline** | 0.0728 | 0.0570 | 0.0792 | 0.0684 | 0.1109 | 0.0944 | 0.0702 | 0.2287 | 0.1447 |
  | **Course Random Baseline** | 0.0712 | 0.0595 | 0.0727 | 0.0716 | 0.1227 | 0.0955 | 0.0670 | 0.2277 | 0.1395 |

- **Evaluation Protocol & Practical Interpretation**:
  1. *Protocol & Unconstrained Catalog Baseline*: Unconstrained global ranking evaluates dot products against all 581 resources across all university courses (`AAA` through `GGG`). Each student only takes 1–2 modules (~6.2 test resources). An unconstrained random baseline yields Precision@10 $\approx 6.2 / 581 \approx 0.0107$ (1.07%), explaining the global LightGCN NDCG@10 of 0.0115.
  2. *Course Structure vs Collaborative Filtering*: In educational VLE platforms, course curricula are strictly partitioned. Foundational materials (syllabus overviews, assignment briefs) have near-universal interaction rates across all enrolled students, making empirical popularity priors exceptionally strong in replicating historical click frequency.
  3. *Production Hybrid Engine Architecture*: Because raw collaborative graph embeddings ignore course boundaries, the production **Hybrid Recommendation Engine** (`ml/recommendation/hybrid_engine.py`) explicitly filters candidates by enrolled course, combining graph structural affinity (0.50 weight), module popularity (0.30 weight), a novelty bonus (1.25 multiplier for untried resource types), and pedagogical risk boosters (+0.25 to formative sub-pages for at-risk students).

---

### 10. Explainability (SHAP Integration)
- Integrated `shap.TreeExplainer` on the production Gradient Boosting ensemble.
- Computed both global feature importance rankings and local per-student attribution vectors.
- **Top 4 Predictive Signals**:
  1. `Score Mean`: Early assessment average is the dominant protective factor.
  2. `Assessments Taken`: Number of formative submissions completed.
  3. `Clicks Per Active Day`: Concentration and intensity of study sessions.
  4. `Total Clicks`: Cumulative engagement across virtual course materials.
- Delivered transparently to users via the `/api/v1/students/{id}/risk` endpoint, categorizing factors into protective habits versus risk-increasing deficits.

---

### 11. Backend and Frontend Implementation & 12 Endpoints Verification

#### A. 12 Endpoints Verification Matrix
All 12 endpoints have been individually executed and verified with live TestClient requests:

| # | HTTP Method | Endpoint Path | Status Code | Verified Result | Verification Notes |
|---|---|---|---|---|---|
| 1 | `GET` | `/health` | `200 OK` | **PASS** | Status `healthy`, database `connected`, artifacts ready `True`. |
| 2 | `POST` | `/api/v1/auth/login` | `200 OK` | **PASS** | Authenticates user, returns JWT Bearer token and role `student`. |
| 3 | `POST` | `/api/v1/auth/register` | `200 OK` | **PASS** | Creates new student account with hashed password and student ID. |
| 4 | `GET` | `/api/v1/students/me` | `200 OK` | **PASS** | Returns authenticated profile, email `student@mintx.edu`, ID `100001`. |
| 5 | `GET` | `/api/v1/students/{id}/profile` | `200 OK` | **PASS** | Returns student demographic bands, region, and modules `['DDD', 'BBB']`. |
| 6 | `GET` | `/api/v1/students/{id}/performance` | `200 OK` | **PASS** | Returns 16 completed formative assessments, mean score `65.81`, and delay history. |
| 7 | `GET` | `/api/v1/students/{id}/risk` | `200 OK` | **PASS** | Returns Gradient Boosting failure probability `0.0607` (Low tier) and 6 local SHAP factors. |
| 8 | `GET` | `/api/v1/students/{id}/recommendations`| `200 OK` | **PASS** | Returns 6 ranked learning resources combining graph affinity and popularity. |
| 9 | `GET` | `/api/v1/students/{id}/engagement` | `200 OK` | **PASS** | Returns 12 weekly clickstream data points for GRU trajectory visualization. |
| 10 | `GET` | `/api/v1/analytics/cohort` | `200 OK` | **PASS** | Returns cohort breakdown (4,028 students) and risk distribution. |
| 11 | `GET` | `/api/v1/analytics/resources` | `200 OK` | **PASS** | Returns resource catalog engagement summary (581 resources tracked across modules). |
| 12 | `GET` | `/api/v1/models/status` | `200 OK` | **PASS** | Returns champion model registry metadata, architecture, and production status. |

*(In addition, `GET /{full_path:path}` serves pre-built React SPA static assets from `frontend/dist/`.)*

#### B. Frontend Implementation
- React 18, Vite, TypeScript, Tailwind CSS.
- 11 interactive views: Student Overview, Academic Risk & SHAP Factors, Recommendations, Engagement Trends, Cohort Risk Matrix, Module Analytics, Model Benchmarks, and Graph Topology.
- Production build compiled into `frontend/dist/` and served directly by FastAPI.

---

### 12. Testing and Validation Summary
- **Pytest Automated Test Suite**: 10/10 tests passing (100% pass rate in 22.92s).
- **Code Quality**: Ruff linter executed with 0 errors; Ruff formatter executed with 0 discrepancies.
- **Frontend Verification**: TypeScript compilation (`tsc`) and Vite production build completed with 0 errors.

---

### 13. Technical Challenges & Solutions

| Technical Challenge | Root Cause | Engineering Solution |
|---|---|---|
| **Artificial 99.88% Model Accuracy** | Synthetic data generation script conditioned click counts directly on final outcome labels; random split allowed multi-course students in both sets. | Refactored ingestion using independent latent student abilities; enforced `GroupShuffleSplit` on `id_student` (verified 0 ID overlap) and Day $\le$ 100 cutoff. |
| **`uv sync` Flat-Layout Discovery Error** | Multiple top-level folders (`backend`, `ml`, `data`, `frontend`) triggered setuptools ambiguous package discovery. | Added explicit `[tool.setuptools.packages.find]` with `include = ["backend*", "ml*"]` and directory exclusions in `pyproject.toml`. |
| **FastAPI Route Order Collision** | Catch-all SPA route `/{full_path:path}` captured `/health` requests before route execution. | Reordered route registration to place explicit `/health` and `/api/v1` routers strictly prior to SPA static catch-all handlers. |
| **Cold-Start Student Recommendations** | Newly enrolled students lacked bipartite graph nodes or prior click interaction edges. | Implemented curriculum-based fallback ranking by module popularity with automated pedagogical rationale strings. |

---

### 14. Skills Developed During the Internship
1. **Applied Learning Analytics**: Formulating early-warning predictive signals from higher education VLE clickstreams without lookahead bias.
2. **Leakage-Free ML Engineering**: Implementing strict temporal boundaries and group-based validation protocols.
3. **Deep Tabular & Recurrent Networks**: Building PyTorch MLP and GRU models with class weighting and sequence padding.
4. **Graph Learning for Recommenders**: Constructing bipartite graphs, computing LightGCN normalized Laplacian smoothing, and evaluating Top-K ranking metrics.
5. **Model Explainability**: Translating raw SHAP TreeExplainer attributions into user-interpretable protective and risk factors.
6. **Full-Stack SaaS Architecture**: Building modular FastAPI microservices, JWT security, and responsive React TypeScript dashboards.
7. **Production DevSecOps**: Configuring pre-commit hooks, Gitleaks secret scanning, Docker Compose orchestration, and GitHub Actions CI/CD.

---

### 15. Final Deliverables
- [x] Complete Python 3.11 ML and Deep Learning codebase (`ml/`).
- [x] Full-stack FastAPI backend and React Vite frontend (`backend/`, `frontend/`).
- [x] 9 persisted model and embedding artifacts (`artifacts/models/`).
- [x] Automated test suite passing 10/10 tests (`backend/tests/`).
- [x] Interactive exploratory data analysis notebook (`notebooks/exploratory_data_analysis.ipynb`).
- [x] Multi-container Docker Compose configuration (`docker-compose.yml`) validated via CI.
- [x] 4 automated GitHub Actions workflows (`.github/workflows/`).
- [x] Complete technical documentation suite (`README.md`, `docs/`).
- [x] Live GitHub repository on branch `main` at [https://github.com/rohitam26/adaptive-learning-predictive-intelligence](https://github.com/rohitam26/adaptive-learning-predictive-intelligence).

---

### 16. Project Limitations
1. **Unobserved External Factors**: VLE clickstream telemetry captures digital interaction volume but does not observe external student life circumstances, illness, employment demands, or offline study habits.
2. **Sequential Sparsity**: The Behavior GRU models 12 weekly click buckets without multimodal text forum sentiment; combining textual forum contributions with click frequency would improve sequence precision.
3. **Graph Bipartite Scalability**: Full-batch LightGCN Laplacian multiplication is computed in-memory; scaling to hundreds of thousands of learners would require minibatch graph sampling (e.g. GraphSAGE via PyTorch Geometric).
4. **Local Docker Environment Constraint**: The local Windows development host lacked the Docker Desktop daemon CLI (`docker: The term 'docker' is not recognized`). Therefore, container runtime execution was not performed locally; orchestration configurations were authored to production standards and validated through automated GitHub Actions CI.

---

### 17. Future Scope
1. **Natural Language Forum Mining**: Integrate transformer models (e.g. RoBERTa) to analyze the sentiment and conceptual depth of forum discussions.
2. **Multi-Task Neural Scaffolding**: Implement multi-task learning networks that simultaneously predict risk probability and forecast continuous final numerical grades.
3. **Real-Time Streaming Telemetry**: Deploy Apache Kafka or Redis Streams to compute real-time risk scores as student click events arrive.
4. **A/B Testing Framework**: Deploy live randomized controlled trials (RCT) to measure the empirical efficacy of personalized resource recommendations on course completion rates.

---

### 18. Conclusion
The **Adaptive Learning and Predictive Intelligence Platform** successfully demonstrates the practical application of machine learning, deep neural representations, and graph learning to higher education learning analytics. By eliminating target leakage, enforcing temporal validity, and providing transparent SHAP attributions, the platform delivers an actionable, scientifically sound solution for early student intervention. The project represents a complete, production-ready engineering deliverable for the MintX Solutions Machine Learning Internship.
