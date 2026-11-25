# Adaptive Learning & Predictive Intelligence Platform
### MintX Solutions • Educational Intelligence Engineering Portfolio

[![FastAPI](https://img.shields.io/badge/FastAPI-0.165-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-red.svg)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-5.4-purple.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4-cyan.svg)](https://tailwindcss.com)

An integrated educational intelligence platform designed to analyze learner behavior, forecast academic risk using early diagnostic windows, capture sequential interaction traces via recurrent neural networks, and deliver personalized study recommendations through bipartite graph representations.

---

## 1. Project Identity & Historical Record
- **Official Title**: Adaptive Learning and Predictive Intelligence Platform
- **Organization**: MintX Solutions
- **Role**: Machine Learning Intern
- **Domain**: Educational Technology (EdTech)
- **Historical Internship Period**: 15 June 2026 – 14 July 2026 (30 Days)
- **Workspace**: `E:\develop\adaptive-learning-platform`
- **Proposed Repository**: `adaptive-learning-predictive-intelligence`

---

## 2. Key Capabilities
1. **Early Academic Risk Forecasting**: Evaluates students up to day 100 with zero temporal lookahead bias and group-partitioned splits, achieving **0.7164 F1** and **0.8865 ROC-AUC** (Gradient Boosting Champion).
2. **Deep Tabular & Sequential Modeling**: PyTorch Tabular MLP achieves **0.7357 F1** and **0.8942 ROC-AUC**; Behavior GRU models 12-week click trajectories for sensitivity to early disengagement.
3. **Bipartite Graph Recommendations**: 4,081-node heterogeneous graph with LightGCN-style multi-layer neighborhood aggregation for personalized resource ranking.
4. **Transparent Explainability (SHAP)**: TreeExplainer attributions clarify risk versus protective factors for individual students and cohort administrators.
5. **Multi-Persona SaaS Experience**: Dedicated student, educator, and administrator dashboards built with React, Vite, and Tailwind CSS.

---

## 3. Technology Stack
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP, NumPy, Pandas, SciPy, Joblib
- **Deep Learning**: PyTorch (Tabular MLP and Behavior GRU)
- **Graph Learning**: NetworkX, SciPy Sparse Laplacian, LightGCN representations
- **Backend & API**: Python 3.11, FastAPI, Pydantic, SQLAlchemy, JWT Security, Uvicorn
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts
- **Database**: SQLite (local zero-dependency development) & PostgreSQL (production containerized)
- **Tooling & QA**: `uv`, Pytest, Ruff, Docker, Docker Compose, GitHub Actions

---

## 4. Local Execution Guide (PowerShell)

### Prerequisites
- Python 3.11 with `.venv` installed
- Node.js 18+ and npm installed

### Step 1: Start the Backend Service
```powershell
# Navigate to project root
cd E:\develop\adaptive-learning-platform

# Run FastAPI backend with Uvicorn
.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
*The backend automatically serves both the REST API at `http://127.0.0.1:8000/api/v1` and the built React frontend at `http://127.0.0.1:8000/`.*

### Step 2: Start the Frontend in Development Mode (Optional)
```powershell
# In a separate terminal window
cd E:\develop\adaptive-learning-platform\frontend
npm run dev
```
*Frontend dev server runs at `http://localhost:5173/`.*

### Step 3: Run with Docker Compose (PostgreSQL + Backend + Frontend)
```powershell
cd E:\develop\adaptive-learning-platform
docker-compose up --build
```

---

## 5. Automated Testing & Pipelines
```powershell
# Run full automated test suite (10/10 passing)
.venv\Scripts\pytest.exe -v

# Run code style & linting checks (0 errors)
.venv\Scripts\ruff.exe check backend ml

# Re-run reproducible ML & Deep Learning pipelines
.venv\Scripts\python.exe ml/models/traditional_ml.py
.venv\Scripts\python.exe ml/models/deep_learning.py
.venv\Scripts\python.exe ml/graph_learning/bipartite_graph.py
.venv\Scripts\python.exe ml/evaluation/explainability.py
```

---

## 6. Pre-Configured Demo Accounts

| Role | Email | Password | Primary Interface |
|---|---|---|---|
| **Student** | `student@mintx.edu` | `Pass123!` | Learner Overview, Risk & SHAP, Hybrid Recommendations |
| **Educator** | `educator@mintx.edu` | `Pass123!` | Cohort Analytics, Risk Segmentation, Intervention Tracking |
| **Admin** | `admin@mintx.edu` | `Pass123!` | System Models, Graph Status, Performance Audit |
