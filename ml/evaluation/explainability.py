"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Computes:
1. Global feature importance (mean absolute SHAP values across cohort).
2. Local student-level prediction explanations (individual risk factors).
3. Risk confidence intervals and decision factor attributions.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


class StudentRiskExplainer:
    def __init__(
        self, artifacts_dir: Path = Path("artifacts"), processed_dir: Path = Path("data/processed")
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"

        with open(self.processed_dir / "dataset_metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_columns"]

        # Load XGBoost or Random Forest model for robust TreeExplainer support
        if (self.models_dir / "xgboost.joblib").exists():
            self.model = joblib.load(self.models_dir / "xgboost.joblib")
        elif (self.models_dir / "random_forest.joblib").exists():
            self.model = joblib.load(self.models_dir / "random_forest.joblib")
        else:
            self.model = joblib.load(self.models_dir / "champion_risk_model.joblib")

        test_df = pd.read_csv(self.processed_dir / "test.csv")
        self.X_background = test_df[self.feature_cols].copy()

        self.explainer = shap.TreeExplainer(self.model)

    def compute_global_importance(self, max_samples: int = 300) -> dict[str, Any]:
        sample = self.X_background.iloc[:max_samples]
        shap_values = self.explainer.shap_values(sample)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs = np.mean(np.abs(sv), axis=0)
        sorted_indices = np.argsort(-mean_abs)

        importance_ranking = []
        for idx in sorted_indices:
            feat = self.feature_cols[idx]
            val = float(mean_abs[idx])
            importance_ranking.append(
                {
                    "feature": feat,
                    "importance": round(val, 4),
                    "display_name": feat.replace("_", " ").title(),
                }
            )

        summary = {
            "top_global_features": importance_ranking[:12],
            "baseline_value": float(np.mean(self.explainer.expected_value))
            if hasattr(self.explainer.expected_value, "__iter__")
            else float(self.explainer.expected_value),
        }

        with open(self.metrics_dir / "global_shap_importance.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def explain_student_risk(self, student_features: dict) -> dict[str, Any]:
        row_vals = [student_features.get(col, 0.0) for col in self.feature_cols]
        df_row = pd.DataFrame([row_vals], columns=self.feature_cols)

        shap_val = self.explainer.shap_values(df_row)
        if isinstance(shap_val, list):
            sv = shap_val[1][0]
        elif len(shap_val.shape) == 3:
            sv = shap_val[0, :, 1]
        elif len(shap_val.shape) == 2:
            sv = shap_val[0]
        else:
            sv = shap_val

        prob = float(self.model.predict_proba(df_row)[0, 1])

        contributions = []
        for feat, val, s in zip(self.feature_cols, row_vals, sv):
            s_val = float(s)
            contributions.append(
                {
                    "feature": feat,
                    "display_name": feat.replace("_", " ").title(),
                    "feature_value": round(float(val), 2),
                    "shap_value": round(s_val, 4),
                    "impact": "Increases Risk" if s_val > 0 else "Protective Factor",
                    "magnitude": round(abs(s_val), 4),
                }
            )

        contributions.sort(key=lambda x: x["magnitude"], reverse=True)
        risk_tier = "Critical" if prob >= 0.70 else "Elevated" if prob >= 0.40 else "Low"

        return {
            "predicted_risk_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "confidence": round(abs(prob - 0.5) * 2, 3),
            "top_contributing_factors": contributions[:6],
            "limitations": [
                "Early prediction is evaluated on engagement data up to day 100.",
                "Predictions reflect statistical likelihood and should assist, not substitute, educator discretion.",
                "Extenuating life circumstances external to VLE telemetry are unobserved.",
            ],
        }


if __name__ == "__main__":
    explainer = StudentRiskExplainer()
    global_imp = explainer.compute_global_importance()
    print(
        "Top global features:", [f["display_name"] for f in global_imp["top_global_features"][:4]]
    )
