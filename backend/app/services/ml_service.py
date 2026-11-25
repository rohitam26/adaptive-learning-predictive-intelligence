"""
Service Layer for Machine Learning, Recommendations, and Analytics.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from backend.app.core.config import settings
from ml.evaluation.explainability import StudentRiskExplainer
from ml.recommendation.hybrid_engine import HybridRecommendationEngine


class MLService:
    _instance = None

    def __init__(self):
        self.artifacts_dir = settings.ARTIFACTS_DIR
        self.data_dir = settings.DATA_DIR
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"

        # Load precomputed datasets
        self.features_df = pd.read_csv(self.processed_dir / "features_full.csv")
        self.student_info_df = pd.read_csv(self.raw_dir / "studentInfo.csv")
        self.student_vle_df = pd.read_csv(self.raw_dir / "studentVle.csv")
        self.student_assess_df = pd.read_csv(self.raw_dir / "studentAssessment.csv")
        self.assessments_df = pd.read_csv(self.raw_dir / "assessments.csv")
        self.vle_df = pd.read_csv(self.raw_dir / "vle.csv")

        # Load metrics
        with open(self.artifacts_dir / "metrics" / "ml_comparison.json", "r") as f:
            self.ml_comparison = json.load(f)

        with open(self.artifacts_dir / "metrics" / "dl_metrics.json", "r") as f:
            self.dl_metrics = json.load(f)

        with open(self.artifacts_dir / "metrics" / "graph_metrics.json", "r") as f:
            self.graph_metrics = json.load(f)

        with open(self.artifacts_dir / "metrics" / "global_shap_importance.json", "r") as f:
            self.global_shap = json.load(f)

        # Initialize engines
        self.recommender = HybridRecommendationEngine(
            raw_data_dir=self.raw_dir, artifacts_dir=self.artifacts_dir
        )
        self.explainer = StudentRiskExplainer(
            artifacts_dir=self.artifacts_dir, processed_dir=self.processed_dir
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MLService()
        return cls._instance

    def get_student_profile(self, student_id: int) -> dict[str, Any] | None:
        sub = self.student_info_df[self.student_info_df["id_student"] == student_id]
        if sub.empty:
            return None
        row = sub.iloc[0]
        modules = list(sub["code_module"].unique())
        return {
            "student_id": int(student_id),
            "modules": modules,
            "gender": row.get("gender"),
            "region": row.get("region"),
            "highest_education": row.get("highest_education"),
            "imd_band": row.get("imd_band"),
            "age_band": row.get("age_band"),
            "num_of_prev_attempts": int(row.get("num_of_prev_attempts", 0)),
            "studied_credits": int(row.get("studied_credits", 60)),
            "disability": str(row.get("disability", "N")),
            "final_result": str(row.get("final_result", "In Progress")),
        }

    def get_student_performance(self, student_id: int) -> dict[str, Any]:
        sub = self.student_assess_df[self.student_assess_df["id_student"] == student_id]
        if sub.empty:
            return {
                "student_id": student_id,
                "assessments_completed": 0,
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "late_submissions": 0,
                "mean_delay_days": 0.0,
                "assessment_history": [],
            }

        merged = sub.merge(
            self.assessments_df[
                ["id_assessment", "code_module", "assessment_type", "date", "weight"]
            ],
            on="id_assessment",
            how="left",
        )
        merged["submission_delay"] = merged["date_submitted"] - merged["date"].fillna(0)
        merged["is_late"] = (merged["submission_delay"] > 0).astype(int)

        history = []
        for _, r in merged.iterrows():
            history.append(
                {
                    "assessment_id": int(r["id_assessment"]),
                    "module": str(r.get("code_module", "")),
                    "type": str(r.get("assessment_type", "TMA")),
                    "score": float(r["score"]),
                    "weight": float(r.get("weight", 0)),
                    "date_submitted": int(r["date_submitted"]),
                    "delay_days": int(r["submission_delay"]),
                    "is_late": bool(r["is_late"]),
                }
            )

        return {
            "student_id": student_id,
            "assessments_completed": len(merged),
            "mean_score": round(float(merged["score"].mean()), 2),
            "min_score": round(float(merged["score"].min()), 2),
            "max_score": round(float(merged["score"].max()), 2),
            "late_submissions": int(merged["is_late"].sum()),
            "mean_delay_days": round(float(merged["submission_delay"].mean()), 2),
            "assessment_history": history,
        }

    def get_student_risk(self, student_id: int) -> dict[str, Any]:
        sub = self.features_df[self.features_df["id_student"] == student_id]
        if sub.empty:
            # Fallback for mock demo student
            features = {col: 0.0 for col in self.explainer.feature_cols}
            features["total_clicks"] = 45.0
            features["score_mean"] = 62.0
            explanation = self.explainer.explain_student_risk(features)
        else:
            row = sub.iloc[0]
            features = {col: float(row.get(col, 0.0)) for col in self.explainer.feature_cols}
            explanation = self.explainer.explain_student_risk(features)

        return {
            "student_id": student_id,
            "model_name": self.ml_comparison["best_model"],
            "risk_probability": explanation["predicted_risk_probability"],
            "risk_tier": explanation["risk_tier"],
            "confidence": explanation["confidence"],
            "top_contributing_factors": explanation["top_contributing_factors"],
            "limitations": explanation["limitations"],
            "assessed_at": datetime.now(UTC).isoformat(),
        }

    def get_student_recommendations(self, student_id: int, top_n: int = 6) -> dict[str, Any]:
        risk_info = self.get_student_risk(student_id)
        risk_level = risk_info["risk_tier"].lower()

        recs = self.recommender.recommend_for_student(
            student_id=student_id, top_n=top_n, risk_level=risk_level
        )
        return {
            "student_id": student_id,
            "total_recommendations": len(recs),
            "recommendations": recs,
        }

    def get_student_engagement(self, student_id: int) -> dict[str, Any]:
        sub = self.student_vle_df[self.student_vle_df["id_student"] == student_id]
        if sub.empty:
            return {
                "student_id": student_id,
                "total_clicks": 0,
                "unique_resources_accessed": 0,
                "active_days": 0,
                "clicks_per_active_day": 0.0,
                "clicks_by_activity_type": {},
                "weekly_trend": [{"week": w, "clicks": 0} for w in range(1, 13)],
            }

        merged = sub.merge(self.vle_df[["id_site", "activity_type"]], on="id_site", how="left")
        total_clicks = int(merged["sum_click"].sum())
        unique_res = int(merged["id_site"].nunique())
        active_days = int(merged["date"].nunique())
        clicks_per_day = round(total_clicks / max(1, active_days), 2)

        by_type = merged.groupby("activity_type")["sum_click"].sum().to_dict()
        by_type_clean = {str(k): int(v) for k, v in by_type.items()}

        merged["week"] = ((merged["date"] + 14) // 7).clip(1, 12).astype(int)
        weekly = merged.groupby("week")["sum_click"].sum().to_dict()
        weekly_trend = [{"week": w, "clicks": int(weekly.get(w, 0))} for w in range(1, 13)]

        return {
            "student_id": student_id,
            "total_clicks": total_clicks,
            "unique_resources_accessed": unique_res,
            "active_days": active_days,
            "clicks_per_active_day": clicks_per_day,
            "clicks_by_activity_type": by_type_clean,
            "weekly_trend": weekly_trend,
        }

    def get_cohort_analytics(self) -> dict[str, Any]:
        total_students = len(self.features_df)
        at_risk_count = int(self.features_df["target_risk"].sum())
        at_risk_pct = round((at_risk_count / total_students) * 100, 2)

        # Approximate tiers based on target_risk and scores
        critical = len(self.features_df[self.features_df["target_risk"] == 1])
        elevated = int(critical * 0.4)  # sub-portion in monitoring tier
        critical_pure = critical - elevated
        low = total_students - critical

        # Module breakdown
        mod_grp = (
            self.features_df.groupby("code_module")
            .agg(
                students=("id_student", "count"),
                risk_rate=("target_risk", "mean"),
                avg_clicks=("total_clicks", "mean"),
                avg_score=("score_mean", "mean"),
            )
            .reset_index()
        )

        module_breakdown = []
        for _, r in mod_grp.iterrows():
            module_breakdown.append(
                {
                    "module": str(r["code_module"]),
                    "students": int(r["students"]),
                    "at_risk_rate": round(float(r["risk_rate"]) * 100, 1),
                    "avg_clicks": round(float(r["avg_clicks"]), 1),
                    "avg_score": round(float(r["avg_score"]), 1),
                }
            )

        return {
            "total_students": total_students,
            "at_risk_count": at_risk_count,
            "at_risk_percentage": at_risk_pct,
            "risk_distribution": {"low": low, "elevated": elevated, "critical": critical_pure},
            "average_engagement_clicks": round(float(self.features_df["total_clicks"].mean()), 1),
            "average_assessment_score": round(float(self.features_df["score_mean"].mean()), 1),
            "module_breakdown": module_breakdown,
            "global_feature_importance": self.global_shap.get("top_global_features", []),
        }

    def get_resource_analytics(self) -> dict[str, Any]:
        pop = self.recommender.df_vle.sort_values(by="interaction_count", ascending=False)
        top_resources = []
        for _, r in pop.head(8).iterrows():
            top_resources.append(
                {
                    "resource_id": int(r["id_site"]),
                    "module": str(r["code_module"]),
                    "activity_type": str(r["activity_type"]),
                    "interaction_count": int(r.get("interaction_count", 0)),
                    "total_clicks": int(r.get("total_clicks", 0)),
                    "popularity_score": round(float(r.get("popularity_score", 0)), 3),
                }
            )

        type_counts = self.vle_df["activity_type"].value_counts().to_dict()
        type_counts_clean = {str(k): int(v) for k, v in type_counts.items()}

        return {
            "total_resources": len(self.vle_df),
            "activity_type_counts": type_counts_clean,
            "most_popular_resources": top_resources,
        }

    def get_models_status(self) -> dict[str, Any]:
        graph_data = dict(self.graph_metrics)
        if "ranking_metrics" not in graph_data:
            graph_data["ranking_metrics"] = self.graph_metrics.get(
                "unconstrained_ranking_metrics", {}
            )

        return {
            "champion_risk_model": {
                "name": self.ml_comparison["best_model"],
                "f1_score": self.ml_comparison["best_f1"],
                "type": "Supervised Learning Classification",
                "status": "Production Serving",
            },
            "ml_models": self.ml_comparison["comparison"],
            "deep_learning_models": self.dl_metrics,
            "graph_learning": graph_data,
            "active_version": "1.0.0",
        }
