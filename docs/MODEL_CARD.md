# Model Card
## Adaptive Learning & Predictive Intelligence Models

### 1. Model Details
- **Developing Organization**: MintX Solutions Machine Learning Engineering
- **Model Release Date**: July 2026 / Verified November 2026
- **Model Type**: Supervised Ensemble Classifiers, PyTorch Deep Neural Networks, and Bipartite Graph Embeddings.
- **Artifact Formats**: `.joblib` (Scikit-Learn/XGBoost), `.pt` (PyTorch state dicts), `.npz` (Graph embeddings).

### 2. Intended Use
- **Primary Use Case**: Early identification of learner disengagement and academic failure risk to enable timely educational scaffolding.
- **Secondary Use Case**: Personalized ranking and educational rationale delivery for virtual learning resources.
- **Out-of-Scope Use**: Automated punitive grading, disciplinary tracking, or exclusion from degree qualification.

### 3. Model Architecture & Hyperparameters
- **Champion Supervised Model**: Logistic Regression (C=1.0, max_iter=1000) & XGBoost (n_estimators=120, max_depth=5, lr=0.08, logloss).
- **Deep Tabular MLP**: 2 hidden layers (64, 32), BatchNorm1d, Dropout (0.3), AdamW (lr=0.003).
- **Sequential Behavior GRU**: 2-layer Recurrent GRU (hidden_dim=32), sequence length 12 weeks, Adam (lr=0.001) with gradient clipping norm 1.0.
- **LightGCN Bipartite Embedding**: Dimension 32, 2-hop symmetric Laplacian propagation: $D^{-1/2} A D^{-1/2}$.

### 4. Evaluation Benchmark Metrics (Holdout Test Split $N=808$)
- **Supervised Risk Models**:
  - Accuracy: $0.9988$
  - Precision: $1.0000$
  - Recall: $0.9970$
  - F1-Score: $0.9985$
  - ROC-AUC: $1.0000$
  - PR-AUC: $1.0000$
  - Brier Score: $0.0013$
- **PyTorch Deep Learning**:
  - Tabular MLP: Accuracy $0.9950$, F1-Score $1.0000$, ROC-AUC $1.0000$
  - Behavior GRU: Accuracy $0.4790$, F1-Score $0.6477$, ROC-AUC $0.5000$
- **Bipartite Graph Ranking (Top-K)**:
  - Precision@5: $0.0104$, Recall@5: $0.0112$, NDCG@5: $0.0139$
  - Precision@10: $0.0112$, Recall@10: $0.0255$, NDCG@10: $0.0192$
  - Precision@20: $0.0090$, Recall@20: $0.0401$, NDCG@20: $0.0248$
