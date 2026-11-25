# Model Card
## Adaptive Learning & Predictive Intelligence Models

### 1. Model Details
- **Developing Organization**: MintX Solutions Machine Learning Engineering
- **Model Release Date**: July 2026 / Verified November 2026
- **Model Status**:
  - **Production Champion Predictive Model**: Gradient Boosting Classifier (`champion_risk_model.joblib`)
  - **Production Alternative (XAI)**: XGBoost Classifier (`xgboost.joblib`)
  - **Deep Tabular Representation Model**: PyTorch Tabular MLP (`tabular_mlp.pt`)
  - **Experimental Sequential Model**: PyTorch Behavior GRU (`behavior_gru.pt`)
  - **Structural Graph Representation**: LightGCN Embeddings (`graph_embeddings.npz`)
  - **Production Recommendation Serving**: Hybrid Recommendation Engine (`ml/recommendation/hybrid_engine.py`)

### 2. Intended Use
- **Primary Use Case**: Early identification of learner disengagement and academic failure risk within the first 100 days of semester to enable timely educational scaffolding.
- **Secondary Use Case**: Personalized ranking and educational rationale delivery for virtual learning resources (VLE activities).
- **Out-of-Scope Use**: Automated punitive grading, disciplinary tracking, or exclusion from degree qualification.

### 3. Model Architecture & Hyperparameters
- **Champion Supervised Model**: Gradient Boosting Classifier (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`, `random_state=42`).
- **Ensemble Alternative**: XGBoost Classifier (`n_estimators=100`, `learning_rate=0.08`, `max_depth=4`, `subsample=0.85`, `eval_metric="logloss"`).
- **Deep Tabular MLP**: 2 hidden layers (64, 32), BatchNorm1d, Dropout (0.3), AdamW (`lr=0.003`, `weight_decay=1e-4`), class-weighted BCE loss (`pos_weight=2.76`).
- **Experimental Behavior GRU**: 2-layer Recurrent GRU (`hidden_dim=32`), 12-week click frequency sequence, validation-tuned threshold ($t=0.4215$).
- **LightGCN Bipartite Embedding**: Dimension 32, 2-hop symmetric Laplacian propagation: $D^{-1/2} A D^{-1/2}$.

### 4. Evaluation Benchmark Metrics (Holdout Group Test Split $N=815$, Zero Leaking Students)
All models evaluated against a strictly student-grouped holdout test split (700 test students, 2,800 train students; 0 student overlap).

| Model Candidate | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Status |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | Multimodal Tabular | **0.8601** | **0.7784** | **0.6636** | **0.7164** | **0.8865** | **0.1064** | **Production Champion** |
| **XGBoost Classifier** | Multimodal Tabular | 0.8638 | 0.8046 | 0.6452 | 0.7161 | 0.8927 | 0.1024 | Tree XAI Model |
| **PyTorch Tabular MLP** | Deep Tabular | 0.8528 | 0.7029 | **0.7742** | **0.7368** | **0.8941** | - | Deep Representation |
| **Random Forest** | Multimodal Tabular | 0.8601 | 0.7943 | 0.6406 | 0.7092 | 0.8853 | 0.1058 | Bagging Candidate |
| **Logistic Regression** | Linear Baseline | 0.8466 | 0.7500 | 0.6359 | 0.6883 | 0.8919 | 0.1043 | Linear Baseline |
| **PyTorch Behavior GRU**| Sequential Time-Series | 0.6417 | 0.3860 | 0.5853 | 0.4652 | 0.6914 | - | **Experimental Model** |

### 5. Behavior GRU Experimental Findings & Baseline Comparison
- **Test Set Distribution ($N=815$)**: Class 0 (Non-Risk): 598 (73.37%), Class 1 (At-Risk): 217 (26.63%).
- **Confusion Matrix ($t=0.4215$)**: `[[TN: 396, FP: 202], [FN: 90, TP: 127]]`.
- **Baseline Comparison**:
  - *Majority Class Baseline* (predict all non-risk): Accuracy = 0.7337, F1 = 0.0000.
  - *Heuristic Clickstream Baseline* (total clicks < median): Accuracy = 0.5914, Precision = 0.3571, Recall = 0.6682, F1 = 0.4655.
  - *Behavior GRU (Tuned)*: Accuracy = 0.6417, Precision = 0.3860, Recall = 0.5853, F1 = 0.4652, ROC-AUC = 0.6914.
- **Finding**: While the sequential GRU captures early temporal engagement trends without demographics (ROC-AUC 0.6914 vs 0.50 random), univariate clickstreams alone lack the precision of multimodal tabular fusion models (F1 0.4652 vs 0.7164). It is retained as an experimental temporal diagnostic.

### 6. Bipartite Graph Recommendation Evaluation (4,081 Nodes, 115,529 Edges)

#### A. Course-Filtered Benchmark (Ranking within Enrolled Course Modules)
| Recommendation Approach | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Precision@20 | Recall@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| **LightGCN (Course-Filtered)** | 0.0664 | 0.0565 | 0.0698 | 0.0628 | 0.1073 | 0.0875 | 0.0590 | 0.1972 | 0.1252 |
| **Popularity (Course-Filtered)** | 0.0728 | 0.0570 | 0.0792 | 0.0684 | 0.1109 | 0.0944 | 0.0702 | 0.2287 | 0.1447 |
| **Random (Course-Filtered)** | 0.0712 | 0.0595 | 0.0727 | 0.0716 | 0.1227 | 0.0955 | 0.0670 | 0.2277 | 0.1395 |

#### B. Recommendation System Strategy & Production Serving
Curricular learning environments are heavily structured: mandatory course milestones and syllabus overviews generate high baseline popularity across all active students. Consequently, pure graph propagation on its own does not outperform popularity priors in offline historical interaction recovery. This empirical finding justifies our **Hybrid Personalization Engine** (`HybridRecommendationEngine`), which combines module filtering, popularity priors (0.30), LightGCN structural similarity (0.50), novelty weighting, and pedagogical risk boosters.
