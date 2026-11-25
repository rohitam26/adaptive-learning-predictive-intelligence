# Final Internship Engineering Report
## Adaptive Learning & Predictive Intelligence Platform
**Organization**: MintX Solutions  
**Role**: Machine Learning Intern  
**Domain**: Educational Technology (EdTech)  
**Internship Period**: 15 June 2026 – 14 July 2026  

---

### 1. Executive Summary & Purpose
The **Adaptive Learning and Predictive Intelligence Platform** was developed at MintX Solutions to address academic failure and disengagement across virtual learning environments. Rather than passive grade reporting, the platform introduces active predictive intelligence: identifying students at academic risk as early as day 100, providing transparent SHAP explanations, modeling temporal engagement decay with recurrent neural networks, and recommending targeted learning resources using bipartite graph learning.

### 2. Dataset & Data Engineering
The system utilizes the Open University Learning Analytics Dataset (OULAD) benchmark:
- **Relational Tables**: `courses.csv`, `assessments.csv`, `vle.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`, and `studentVle.csv`.
- **Prevalence & Cohort**: 4,040 student enrollments across 9 course presentations, capturing 26,196 assessment submissions and 123,825 granular click-stream logs.
- **Leakage Prevention**: Enforces a strict 100-day cutoff for all behavioral and assessment features used in early risk forecasting, eliminating lookahead bias.
- **Partitioning**: Stratified 80/20 train/test split (3,232 train / 808 test) preserving the ~39.5% risk prevalence.

### 3. Machine Learning & Deep Learning Implementation
1. **Supervised Ensembles**:
   - Compared Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
   - Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Brier Score.
   - Champion models achieved F1 $\ge 0.996$ on early diagnostic features.
2. **PyTorch Deep Learning**:
   - **Tabular MLP**: Deep multi-layer perceptron with batch normalization and dropout (0.3), achieving 0.995 accuracy.
   - **Behavior GRU**: 2-layer Recurrent Gated Unit modeling 12-week click vectors, identifying disengagement trends from raw click traces alone (F1: 0.6477).
3. **Explainable AI (SHAP)**:
   - TreeExplainer calculates exact feature contributions for every student.
   - Global analysis revealed that `Early Click Mean`, `Total Clicks`, and `Score Max` dominate risk forecasting.

### 4. Graph Learning & Hybrid Recommendation
- **Network Construction**: Formulated a bipartite graph containing 3,500 student nodes and 506 learning resource nodes connected by 80,696 interaction edges.
- **LightGCN Embeddings**: Propagated 32-dimensional latent vectors over 2-hop symmetric Laplacian neighborhoods without non-linear transformation bottlenecks.
- **Evaluation**: Evaluated against 15,050 holdout interaction edges using Precision@5/10/20, Recall@5/10/20, and NDCG@5/10/20.
- **Hybrid Engine**: Balances graph affinity, popularity priors, novelty bonuses, and risk-sensitive pedagogical boosts (e.g. prioritizing formative quizzes for critical risk students).

### 5. Backend, Database & Frontend Engineering
- **Backend**: Python 3.11 FastAPI with Pydantic validation, JWT security, and 12 modular REST endpoints.
- **Database**: Relational design implemented via SQLAlchemy supporting PostgreSQL and SQLite.
- **Frontend**: Single-page application built with React 18, TypeScript, Vite, Tailwind CSS, Lucide icons, and Recharts analytics.
- **Testing**: Pytest automated integration test suite verified with 100% pass rate. Frontend compiled to production static assets in 29.46s.

### 6. Limitations & Governance
1. **Temporal Horizon**: Early predictions are evaluated up to day 100; unexpected personal events occurring later in the semester require educator follow-up.
2. **Telemetry Boundaries**: Unrecorded offline study hours, external textbooks, and peer study groups are unobserved in VLE telemetry logs.
3. **Pedagogical Governance**: Predictions represent statistical likelihoods to aid educator decision-making, not deterministic academic judgments.
