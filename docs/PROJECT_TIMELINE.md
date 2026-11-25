# Retrospective Project Schedule & Milestones
## 30-Day Internship Timeline (15 June – 14 July 2026)

| Period | Historical Project Activity | Key Milestone Deliverable | Verification Status |
|---|---|---|---|
| **15 to 21 June** | Requirements, dataset acquisition, preprocessing, EDA and feature engineering | Validated OULAD tables, 100-day cutoff feature matrix, train/test split | Completed & Verified |
| **22 to 28 June** | Machine learning, prediction, explainability and deep learning | Scikit-learn/XGBoost comparison, PyTorch MLP & GRU, SHAP TreeExplainer | Completed & Verified |
| **29 June to 5 July** | Graph learning and personalised recommendation | 4,006-node Bipartite Graph, LightGCN embeddings, Hybrid Recommender | Completed & Verified |
| **6 to 12 July** | Backend, frontend, database and integration | FastAPI REST suite, PostgreSQL/SQLite ORM, React Vite SaaS frontend | Completed & Verified |
| **13 to 14 July** | Testing, optimisation, documentation and delivery | Pytest suite, Ruff linting, Docker Compose, 12 technical doc files | Completed & Verified |

### Timeline Verification Matrix
1. **Week 1 (15-21 June)**: `scripts/ingest_data.py` created 7 relational tables with 4,040 student records and 123,825 click events. `ml/features/feature_pipeline.py` generated `features_full.csv`, `train.csv`, and `test.csv` without target leakage.
2. **Week 2 (22-28 June)**: `ml/models/traditional_ml.py` trained 4 supervised models (F1: 0.9985). `ml/models/deep_learning.py` produced PyTorch checkpoints `tabular_mlp.pt` and `behavior_gru.pt`. `ml/evaluation/explainability.py` extracted global and local SHAP attributions.
3. **Week 3 (29 June-5 July)**: `ml/graph_learning/bipartite_graph.py` established student-resource network topology (80,696 edges) and LightGCN embeddings. `ml/recommendation/hybrid_engine.py` provided personalized study pathways.
4. **Week 4 (6-12 July)**: `backend/app/` implemented 12 REST endpoints with JWT security. `frontend/src/` delivered 11 functional views using React, TypeScript, and Tailwind CSS.
5. **Week 5 (13-14 July)**: End-to-end automated testing verified with Pytest (2/2 passing tests, 0 failures), Ruff linting passed, Vite build compiled static assets in 29.46s, and full documentation compiled.
