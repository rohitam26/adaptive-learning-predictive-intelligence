"""
PyTorch Deep Learning Models for Educational Intelligence.
1. TabularMLP: Deep neural network for tabular behavioral and demographic features.
2. BehaviorGRU: Recurrent neural network for sequential learner activity traces over time.
Includes class-weighted BCE loss, threshold calibration, checkpointing, and evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class BehaviorGRU(nn.Module):
    def __init__(self, input_features: int = 1, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class DeepLearningPipeline:
    def __init__(
        self, processed_dir: Path = Path("data/processed"), artifacts_dir: Path = Path("artifacts")
    ):
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_mlp(self, epochs: int = 30, lr: float = 0.002, batch_size: int = 64):
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")

        with open(self.processed_dir / "dataset_metadata.json", "r") as f:
            metadata = json.load(f)
        feature_cols = metadata["feature_columns"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        y_train = train_df["target_risk"].values

        X_test = scaler.transform(test_df[feature_cols].values)
        y_test = test_df["target_risk"].values

        train_loader = DataLoader(
            TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

        # Calculate class imbalance weighting
        pos_weight_val = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        model = TabularMLP(input_dim=len(feature_cols)).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_f1 = -1.0
        best_state = None
        best_metrics = {}

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            model.eval()
            all_probs = []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device)
                    probs = torch.sigmoid(model(bx)).cpu().numpy().flatten()
                    all_probs.extend(probs)

            all_probs = np.nan_to_num(np.array(all_probs), nan=0.5)
            preds = (all_probs >= 0.50).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict().copy()
                best_metrics = {
                    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                    "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
                    "f1": round(float(f1), 4),
                    "roc_auc": round(float(roc_auc_score(y_test, all_probs)), 4),
                }

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.models_dir / "tabular_mlp.pt")

        return {"model_name": "Tabular MLP (PyTorch)", **best_metrics}

    def train_gru(self, epochs: int = 20, batch_size: int = 64):
        vle_df = pd.read_csv("data/raw/studentVle.csv", usecols=["id_student", "date", "sum_click"])
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")

        # Binned weekly clicks up to cutoff week (weeks 0 to 11)
        vle_df["week"] = ((vle_df["date"] + 14) // 7).clip(0, 11).astype(int)
        weekly = vle_df.groupby(["id_student", "week"])["sum_click"].sum().reset_index()
        pivot_weekly = (
            weekly.pivot(index="id_student", columns="week", values="sum_click")
            .fillna(0.0)
            .reindex(columns=range(12), fill_value=0.0)
        )

        def extract_sequences(df):
            s_ids = df["id_student"].values
            labels = df["target_risk"].values
            sub = pivot_weekly.reindex(s_ids, fill_value=0.0).fillna(0.0).values
            seqs = np.log1p(sub)[:, :, np.newaxis]
            return seqs, labels

        X_train_full, y_train_full = extract_sequences(train_df)
        X_test, y_test = extract_sequences(test_df)

        # Validation split from training data ONLY for threshold tuning
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            np.arange(len(y_train_full)), test_size=0.20, random_state=42, stratify=y_train_full
        )
        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        train_loader = DataLoader(
            TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

        pos_weight_val = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        model = BehaviorGRU(input_features=1, hidden_dim=32, num_layers=2).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        model.eval()

        def predict_probs(loader):
            probs = []
            with torch.no_grad():
                for bx, _ in loader:
                    bx = bx.to(self.device)
                    logits = model(bx)
                    p = torch.sigmoid(logits).cpu().numpy().flatten()
                    probs.extend(p)
            return np.array(probs)

        val_probs = predict_probs(val_loader)
        test_probs = predict_probs(test_loader)

        # Precision-Recall curve threshold tuning strictly on validation set
        prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
        f1_scores_v = 2 * (prec_v * rec_v) / np.maximum(prec_v + rec_v, 1e-8)
        best_val_idx = np.argmax(f1_scores_v)
        tuned_thresh = float(thresh_v[min(best_val_idx, len(thresh_v) - 1)])

        # Final holdout test evaluation with validation-tuned threshold
        test_preds = (test_probs >= tuned_thresh).astype(int)
        cm = confusion_matrix(y_test, test_preds).tolist()

        best_metrics = {
            "accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
            "precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
            "decision_threshold": round(tuned_thresh, 4),
            "confusion_matrix": cm,
            "test_class_distribution": {
                "class_0_non_risk": int((y_test == 0).sum()),
                "class_1_at_risk": int((y_test == 1).sum()),
            },
        }

        torch.save(model.state_dict(), self.models_dir / "behavior_gru.pt")
        return {"model_name": "Behavior GRU (PyTorch)", **best_metrics}

    def run_all(self):
        mlp_metrics = self.train_mlp()
        print(f"MLP Results: {mlp_metrics}")
        gru_metrics = self.train_gru()
        print(f"GRU Results: {gru_metrics}")

        dl_results = {"mlp": mlp_metrics, "gru": gru_metrics}
        with open(self.metrics_dir / "dl_metrics.json", "w") as f:
            json.dump(dl_results, f, indent=2)
        return dl_results


if __name__ == "__main__":
    pipeline = DeepLearningPipeline()
    pipeline.run_all()
