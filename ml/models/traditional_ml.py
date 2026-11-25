"""
Machine Learning Training, Comparison and Model Selection.
Evaluates:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
Metrics:
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score, Confusion Matrix.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class MLModelTrainer:
    def __init__(
        self, processed_dir: Path = Path("data/processed"), artifacts_dir: Path = Path("artifacts")
    ):
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(self.processed_dir / "dataset_metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata["feature_columns"]

    def load_data(self):
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")

        X_train = train_df[self.feature_cols].copy()
        y_train = train_df["target_risk"].values

        X_test = test_df[self.feature_cols].copy()
        y_test = test_df["target_risk"].values

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        joblib.dump(scaler, self.models_dir / "scaler.joblib")

        return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled

    def evaluate_model(self, name: str, model, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
        brier = float(brier_score_loss(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
        }

    def train_and_compare(self):
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled = self.load_data()

        models = {
            "Logistic Regression": {
                "instance": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
                "use_scaled": True,
            },
            "Random Forest": {
                "instance": RandomForestClassifier(
                    n_estimators=150, max_depth=12, min_samples_split=4, random_state=42
                ),
                "use_scaled": False,
            },
            "Gradient Boosting": {
                "instance": GradientBoostingClassifier(
                    n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
                ),
                "use_scaled": False,
            },
            "XGBoost": {
                "instance": XGBClassifier(
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=5,
                    eval_metric="logloss",
                    random_state=42,
                ),
                "use_scaled": False,
            },
        }

        results = {}
        best_f1 = -1.0
        best_model_name = ""
        best_model_instance = None

        for name, config in models.items():
            print(f"Training {name}...")
            model = config["instance"]
            X_tr = X_train_scaled if config["use_scaled"] else X_train
            X_te = X_test_scaled if config["use_scaled"] else X_test

            model.fit(X_tr, y_train)
            eval_metrics = self.evaluate_model(name, model, X_te, y_test)
            results[name] = eval_metrics
            print(
                f" -> {name} | F1: {eval_metrics['f1']} | ROC-AUC: {eval_metrics['roc_auc']} | Brier: {eval_metrics['brier_score']}"
            )

            # Persist individual model
            slug = name.lower().replace(" ", "_")
            joblib.dump(model, self.models_dir / f"{slug}.joblib")

            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                best_model_name = name
                best_model_instance = model

        # Save production champion model
        joblib.dump(best_model_instance, self.models_dir / "champion_risk_model.joblib")

        summary = {
            "best_model": best_model_name,
            "best_f1": best_f1,
            "comparison": results,
            "feature_names": self.feature_cols,
        }

        with open(self.metrics_dir / "ml_comparison.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nModel selection complete! Champion: {best_model_name} (F1: {best_f1})")
        return summary


if __name__ == "__main__":
    trainer = MLModelTrainer()
    trainer.train_and_compare()
