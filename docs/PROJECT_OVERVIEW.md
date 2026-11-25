# Adaptive Learning & Predictive Intelligence Platform
## Project Overview

### 1. Executive Summary
The **Adaptive Learning and Predictive Intelligence Platform** is an enterprise-grade educational technology system developed for **MintX Solutions**. The platform analyzes multidimensional learner behavior, forecasts academic outcomes with early warning risk detection, captures sequential study trajectories using deep recurrent architectures, and delivers personalized learning resource recommendations through bipartite graph representations.

### 2. Organizational & Internship Context
- **Organization**: MintX Solutions
- **Role**: Machine Learning Intern
- **Domain**: Educational Technology (EdTech) & Predictive Learning Analytics
- **Historical Internship Period**: 15 June 2026 – 14 July 2026 (30 Days)
- **Proposed Repository**: `adaptive-learning-predictive-intelligence`
- **Primary Workspace**: `E:\develop\adaptive-learning-platform`

### 3. Core Problems Addressed
Traditional virtual learning environments record vast telemetry logs but remain passive, failing to intervene before student attrition occurs. This platform resolves key structural challenges:
1. **Early Academic Attrition**: Identifies students at risk before assessment failure deadlines through early 100-day cutoff evaluation without temporal leakage.
2. **Behavioral Black Boxes**: Couples predictive models with SHAP (SHapley Additive exPlanations) to clarify contributing risk factors versus protective behaviors.
3. **Impersonal Curriculum Navigation**: Employs LightGCN-style bipartite graph neural representations to recommend high-yield learning resources tailored to individual student gaps.
4. **Disjointed Educator Workflows**: Unifies student-level diagnostics and cohort-wide predictive intelligence within a synchronized SaaS interface.

### 4. High-Level System Architecture
The platform is designed around four integrated layers:
- **Data Engineering**: Robust ingestion and validation of the Open University Learning Analytics Dataset (OULAD) across 7 relational tables.
- **Machine Learning Core**: Comparative supervised models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost), PyTorch sequential deep learning (Tabular MLP and Behavior GRU), and NetworkX bipartite graph convolution.
- **Backend Service Layer**: High-performance Python 3.11 FastAPI service with SQLAlchemy ORM, JWT security, and modular REST endpoints.
- **Frontend SaaS Experience**: React 18, TypeScript, Vite, Tailwind CSS, Lucide icons, and Recharts analytics dashboards.
