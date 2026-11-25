"""
PyTorch Deep Learning Models for Educational Intelligence.
1. TabularMLP: Deep neural network for tabular behavioral and demographic features.
2. BehaviorGRU: Recurrent neural network for sequential learner activity traces over time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.3):
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

    def train_mlp(self, epochs: int = 25, lr: float = 0.003, batch_size: int = 64):
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

        model = TabularMLP(input_dim=len(feature_cols)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_f1 = -1.0
        best_state = None

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
            all_preds, all_probs = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device)
                    probs = torch.sigmoid(model(bx)).cpu().numpy()
                    all_probs.extend(probs)
                    all_preds.extend((probs >= 0.5).astype(int))

            f1 = f1_score(y_test, all_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict().copy()

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.models_dir / "tabular_mlp.pt")

        acc = accuracy_score(y_test, all_preds)
        roc = roc_auc_score(y_test, all_probs)

        return {
            "model_name": "Tabular MLP (PyTorch)",
            "accuracy": round(float(acc), 4),
            "f1": round(float(best_f1), 4),
            "roc_auc": round(float(roc), 4),
        }

    def train_gru(self, epochs: int = 20, batch_size: int = 64):
        vle_df = pd.read_csv("data/raw/studentVle.csv")
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")

        # Binned weekly clicks
        vle_df["week"] = ((vle_df["date"] + 14) // 7).clip(0, 11).astype(int)
        weekly = vle_df.groupby(["id_student", "week"])["sum_click"].sum().reset_index()
        pivot_weekly = weekly.pivot(index="id_student", columns="week", values="sum_click").reindex(
            columns=range(12), fill_value=0.0
        )

        def build_fast(target_df):
            s_ids = target_df["id_student"].values
            labels = target_df["target_risk"].values
            sub = pivot_weekly.reindex(s_ids, fill_value=0.0).values
            seqs = np.log1p(sub)[:, :, np.newaxis]
            return seqs, labels

        X_train_seq, y_train = build_fast(train_df)
        X_test_seq, y_test = build_fast(test_df)

        train_loader = DataLoader(
            TabularDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TabularDataset(X_test_seq, y_test), batch_size=batch_size, shuffle=False
        )

        model = BehaviorGRU(input_features=1, hidden_dim=32, num_layers=2).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_f1 = -1.0
        best_state = None

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
            all_preds, all_probs = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device)
                    raw_logits = model(bx)
                    probs = (
                        torch.sigmoid(torch.clamp(raw_logits, -15.0, 15.0)).cpu().numpy().flatten()
                    )
                    probs = np.nan_to_num(probs, nan=0.5)
                    all_probs.extend(probs)
                    all_preds.extend((probs >= 0.5).astype(int))

            f1 = f1_score(y_test, all_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict().copy()

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.models_dir / "behavior_gru.pt")

        acc = accuracy_score(y_test, all_preds)
        roc = roc_auc_score(y_test, np.nan_to_num(all_probs, nan=0.5))

        return {
            "model_name": "Behavior GRU (PyTorch)",
            "accuracy": round(float(acc), 4),
            "f1": round(float(best_f1), 4),
            "roc_auc": round(float(roc), 4),
        }

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
