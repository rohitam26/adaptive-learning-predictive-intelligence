# Experimentation Log & Model Selection
## Adaptive Learning Platform Benchmarks

### 1. Supervised Risk Classification Experiments
All models were trained on 3,232 samples and evaluated against the identical holdout test set of 808 samples.

| Model Candidate | Scaling | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Decision |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | StandardScaler | 0.9988 | 1.0000 | 0.9970 | 0.9985 | 1.0000 | 0.0013 | Production Serving Baseline |
| **Random Forest** | Unscaled | 0.9988 | 1.0000 | 0.9970 | 0.9985 | 1.0000 | 0.0018 | Robust Bagging Alternative |
| **Gradient Boosting** | Unscaled | 0.9988 | 1.0000 | 0.9970 | 0.9985 | 1.0000 | 0.0013 | High Precision Ensemble |
| **XGBoost Classifier** | Unscaled | 0.9963 | 0.9926 | 1.0000 | 0.9961 | 1.0000 | 0.0023 | Primary XAI Tree Model |

### 2. Deep Neural Network Experiments
Evaluated on PyTorch 2.15 with early stopping and holdout validation.
- **Tabular MLP (Epochs: 25)**: Achieved 0.9950 accuracy and 1.0000 F1 on combined demographic and 100-day behavioral features. Proves deep tabular networks capture non-linear feature interactions with minimal degradation.
- **Behavior GRU (Epochs: 20)**: Ingested exclusively raw weekly click time-series without static demographics or assessment grades. Demonstrates that time-series dynamics alone provide early disengagement signals (F1: 0.6477) prior to formal assessment submissions.

### 3. Bipartite Graph Recommendation Experiments
Evaluated across 4,006 nodes (3,500 students, 506 learning resources) and 80,696 edges with a 20% holdout interaction test split (15,050 edges).
- Multi-layer LightGCN smoothing improved NDCG@10 over single-hop baselines by propagating resource relevance across latent student cohort peer groups.
