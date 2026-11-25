"""
Investigation, convergence analysis, and baseline comparison for Behavior GRU.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
vle_df = pd.read_csv("data/raw/studentVle.csv", usecols=["id_student", "date", "sum_click"])

# Precompute weekly click sequences (weeks 0 to 11)
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
    total_clicks = sub.sum(axis=1)
    return seqs, labels, total_clicks


X_train_full, y_train_full, clicks_train_full = extract_sequences(train_df)
X_test, y_test, clicks_test = extract_sequences(test_df)

# Split train_df into train (80%) and validation (20%) for threshold tuning
train_idx, val_idx = train_test_split(
    np.arange(len(y_train_full)), test_size=0.20, random_state=42, stratify=y_train_full
)

X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=64, shuffle=False)


# Architecture
class BehaviorGRU(nn.Module):
    def __init__(self, input_features=1, hidden_dim=32, num_layers=2):
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
        return self.fc(last_step)


device = torch.device("cpu")
model = BehaviorGRU().to(device)

# Class weight calculation
pos_weight_val = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
)
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

# Training loop
epochs = 20
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(by)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

# Inference on Validation Set
model.eval()


def get_probs(loader):
    probs = []
    with torch.no_grad():
        for bx, _ in loader:
            logits = model(bx)
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(p)
    return np.array(probs)


val_probs = get_probs(val_loader)
test_probs = get_probs(test_loader)

# PR Curve and Threshold Tuning on VALIDATION set only
prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
f1_scores_v = 2 * (prec_v * rec_v) / np.maximum(prec_v + rec_v, 1e-8)
best_val_idx = np.argmax(f1_scores_v)
tuned_thresh = float(thresh_v[min(best_val_idx, len(thresh_v) - 1)])

# Holdout Test Evaluation at Tuned Threshold
test_preds = (test_probs >= tuned_thresh).astype(int)
cm = confusion_matrix(y_test, test_preds)

print("=" * 60)
print("BEHAVIOR GRU EXPERIMENTAL EVALUATION")
print("=" * 60)
print(f"Validation Tuned Threshold: {tuned_thresh:.4f}")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds):.4f}")
print(f"Test Precision: {precision_score(y_test, test_preds, zero_division=0):.4f}")
print(f"Test Recall:    {recall_score(y_test, test_preds, zero_division=0):.4f}")
print(f"Test F1:        {f1_score(y_test, test_preds, zero_division=0):.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")
print(f"Test PR-AUC:    {average_precision_score(y_test, test_probs):.4f}")
print("Confusion Matrix:")
print(cm)
print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}, TP: {cm[1, 1]}")

# Baseline Comparisons on Test Set
# 1. Majority Class Baseline (predict 0 for all)
maj_preds = np.zeros_like(y_test)
maj_f1 = f1_score(y_test, maj_preds, zero_division=0)
maj_acc = accuracy_score(y_test, maj_preds)

# 2. Simple Total Clicks Heuristic Baseline (predict at-risk if total clicks < median)
click_thresh = np.median(clicks_train_full)
click_preds = (clicks_test < click_thresh).astype(int)
click_f1 = f1_score(y_test, click_preds, zero_division=0)
click_acc = accuracy_score(y_test, click_preds)
click_rec = recall_score(y_test, click_preds, zero_division=0)
click_prec = precision_score(y_test, click_preds, zero_division=0)

print("\n" + "=" * 60)
print("BASELINE COMPARISONS (TEST SET N=815)")
print("=" * 60)
print(f"Majority Class Baseline:   Accuracy = {maj_acc:.4f}, F1 = {maj_f1:.4f}")
print(
    f"Total Clicks < Median:     Accuracy = {click_acc:.4f}, Precision = {click_prec:.4f}, Recall = {click_rec:.4f}, F1 = {click_f1:.4f}"
)
print(
    f"Behavior GRU (Tuned):      Accuracy = {accuracy_score(y_test, test_preds):.4f}, Precision = {precision_score(y_test, test_preds, zero_division=0):.4f}, Recall = {recall_score(y_test, test_preds, zero_division=0):.4f}, F1 = {f1_score(y_test, test_preds, zero_division=0):.4f}"
)

# Save retrained model artifact
torch.save(model.state_dict(), "artifacts/models/behavior_gru.pt")
print("\nSuccessfully saved behavior_gru.pt")
