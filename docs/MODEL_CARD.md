# Model Card
## Adaptive Learning & Predictive Intelligence Models

### 1. Model Details
- **Developing Organization**: MintX Solutions Machine Learning Engineering
- **Model Release Date**: July 2026 / Verified November 2026
- **Model Type**: Supervised Ensemble Classifiers, PyTorch Deep Neural Networks, and Bipartite Graph Embeddings.
- **Artifact Formats**: `.joblib` (Scikit-Learn/XGBoost), `.pt` (PyTorch state dicts), `.npz` (Graph embeddings).
- **Target Variable**: Early Academic Risk (`target_risk`: 1 for Fail/Withdrawn, 0 for Pass/Distinction).

### 2. Intended Use
- **Primary Use Case**: Early identification of learner disengagement and academic failure risk within the first 100 days of semester to enable timely educational scaffolding.
- **Secondary Use Case**: Personalized ranking and educational rationale delivery for virtual learning resources (VLE activities).
- **Out-of-Scope Use**: Automated punitive grading, disciplinary tracking, or exclusion from degree qualification.

### 3. Model Architecture & Hyperparameters
- **Champion Supervised Model**: Gradient Boosting Classifier (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`, random_state=42).
- **Ensemble Alternative**: XGBoost Classifier (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`, `eval_metric="logloss"`).
- **Deep Tabular MLP**: 2 hidden layers (64, 32), BatchNorm1d, Dropout (0.3), AdamW (`lr=0.003`, `weight_decay=1e-4`), class-weighted BCE loss (`pos_weight=2.76`).
- **Sequential Behavior GRU**: 2-layer Recurrent GRU (`hidden_dim=32`), sequence length 12 weeks, Adam (`lr=0.001`) with gradient norm clipping (1.0).
- **LightGCN Bipartite Embedding**: Dimension 32, 2-hop symmetric Laplacian propagation: $D^{-1/2} A D^{-1/2}$.

### 4. Evaluation Benchmark Metrics (Holdout Group Test Split $N=815$, Zero Leaking Students)
All models evaluated against a strictly student-grouped holdout test split (700 test students, 2,800 train students; 0 student overlap).

| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting (Champion)** | **0.8601** | **0.7784** | **0.6636** | **0.7164** | **0.8865** | **0.8118** | **0.1064** |
| **XGBoost Classifier** | 0.8638 | 0.8046 | 0.6452 | 0.7161 | 0.8927 | 0.8230 | 0.1024 |
| **Random Forest** | 0.8601 | 0.7943 | 0.6406 | 0.7092 | 0.8853 | 0.8054 | 0.1058 |
| **Logistic Regression** | 0.8466 | 0.7500 | 0.6359 | 0.6883 | 0.8919 | 0.8163 | 0.1043 |
| **PyTorch Tabular MLP** | 0.8528 | 0.7046 | 0.7696 | 0.7357 | 0.8942 | - | - |
| **PyTorch Behavior GRU** | 0.2663 | 0.2663 | 1.0000 | 0.4205 | 0.5000 | - | - |

### 5. Bipartite Graph Recommendation Evaluation (4,081 Nodes, 115,529 Edges)
Evaluated on holdout interaction edges ($N=21,845$ test edges, $N=93,684$ train edges):
- **Top-5 Ranking**: Precision@5: 0.0096 | Recall@5: 0.0065 | NDCG@5: 0.0081
- **Top-10 Ranking**: Precision@10: 0.0100 | Recall@10: 0.0160 | NDCG@10: 0.0115
- **Top-20 Ranking**: Precision@20: 0.0108 | Recall@20: 0.0325 | NDCG@20: 0.0189

### 6. Explainability & Feature Attributions
SHAP TreeExplainer attributions identify the dominant predictive indicators:
1. `Score Mean` (Mean assessment score in first 100 days)
2. `Assessments Taken` (Count of formative/summative submissions completed)
3. `Clicks Per Active Day` (Intensity of engagement per active study session)
4. `Total Clicks` (Cumulative clicks on course virtual learning environment)
