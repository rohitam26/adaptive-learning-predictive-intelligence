"""
Hybrid Recommendation Engine for Personalized Educational Resources.
Combines:
1. Graph-based structural affinity (LightGCN graph embeddings)
2. Collaborative filtering interaction patterns
3. Resource engagement & popularity
4. Academic risk context
5. Cold-start fallback mechanisms
Provides ranked recommendations with human-interpretable educational rationales.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class HybridRecommendationEngine:
    def __init__(
        self, raw_data_dir: Path = Path("data/raw"), artifacts_dir: Path = Path("artifacts")
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"

        self.df_vle = pd.read_csv(self.raw_data_dir / "vle.csv")
        self.df_student_vle = pd.read_csv(self.raw_data_dir / "studentVle.csv")
        self.df_student_info = pd.read_csv(self.raw_data_dir / "studentInfo.csv")

        # Load precomputed graph embeddings if available
        emb_path = self.models_dir / "graph_embeddings.npz"
        if emb_path.exists():
            self.embeddings = dict(np.load(emb_path))
        else:
            self.embeddings = {}

        # Compute resource popularity and avg clicks
        pop = (
            self.df_student_vle.groupby("id_site")
            .agg(interaction_count=("id_student", "nunique"), total_clicks=("sum_click", "sum"))
            .reset_index()
        )
        self.df_vle = self.df_vle.merge(pop, on="id_site", how="left").fillna(0)
        self.df_vle["popularity_score"] = self.df_vle["interaction_count"] / (
            self.df_vle["interaction_count"].max() + 1e-6
        )

    def recommend_for_student(
        self, student_id: int, top_n: int = 5, risk_level: str = "medium"
    ) -> list[dict[str, Any]]:
        stud_key = f"stud_{student_id}"

        # Check student enrollment to filter module resources
        student_records = self.df_student_info[self.df_student_info["id_student"] == student_id]
        if student_records.empty:
            modules = ["AAA", "BBB"]
        else:
            modules = list(student_records["code_module"].unique())

        candidate_resources = self.df_vle[self.df_vle["code_module"].isin(modules)].copy()

        # Already interacted items
        past_interactions = self.df_student_vle[self.df_student_vle["id_student"] == student_id]
        interacted_sites = set(past_interactions["id_site"].unique())

        # Cold start check: if student has no graph embedding or no prior clicks
        if stud_key not in self.embeddings or len(interacted_sites) == 0:
            return self._cold_start_recommendations(candidate_resources, top_n, risk_level)

        stud_emb = self.embeddings[stud_key]

        scores = []
        for _, row in candidate_resources.iterrows():
            site_id = int(row["id_site"])
            res_key = f"res_{site_id}"

            # Graph affinity score
            if res_key in self.embeddings:
                res_emb = self.embeddings[res_key]
                dot_prod = float(np.dot(stud_emb, res_emb))
                graph_affinity = 1.0 / (1.0 + np.exp(-dot_prod))
            else:
                graph_affinity = 0.3

            # Activity type booster based on risk
            act_type = str(row["activity_type"])
            pedagogical_boost = 1.0
            if risk_level in ["high", "critical"]:
                # High risk students benefit from formative quizzes and core content
                if act_type in ["quiz", "externalquiz"]:
                    pedagogical_boost = 1.35
                elif act_type in ["oucontent", "resource"]:
                    pedagogical_boost = 1.20
            else:
                # Moderate/low risk students benefit from peer discussion and subpages
                if act_type in ["forumng", "collaborate"]:
                    pedagogical_boost = 1.25
                elif act_type in ["oucontent"]:
                    pedagogical_boost = 1.15

            # Popularity prior
            pop_score = float(row["popularity_score"])

            # Novelty bonus (prefer unvisited items)
            novelty = 1.25 if site_id not in interacted_sites else 0.75

            # Hybrid combined score
            final_score = (
                (0.50 * graph_affinity + 0.30 * pop_score + 0.20) * pedagogical_boost * novelty
            )

            # Rationale explanation
            rationale = self._generate_rationale(act_type, risk_level, site_id in interacted_sites)

            scores.append(
                {
                    "resource_id": site_id,
                    "module": row["code_module"],
                    "activity_type": act_type,
                    "affinity_score": round(final_score, 4),
                    "graph_affinity": round(graph_affinity, 4),
                    "popularity_score": round(pop_score, 4),
                    "has_interacted": site_id in interacted_sites,
                    "rationale": rationale,
                }
            )

        # Sort and return top N
        scores.sort(key=lambda x: x["affinity_score"], reverse=True)
        return scores[:top_n]

    def _cold_start_recommendations(
        self, candidate_resources: pd.DataFrame, top_n: int, risk_level: str
    ):
        ranked = candidate_resources.sort_values(by="popularity_score", ascending=False).head(top_n)
        results = []
        for _, row in ranked.iterrows():
            act_type = str(row["activity_type"])
            results.append(
                {
                    "resource_id": int(row["id_site"]),
                    "module": row["code_module"],
                    "activity_type": act_type,
                    "affinity_score": round(float(row["popularity_score"]), 4),
                    "graph_affinity": 0.50,
                    "popularity_score": round(float(row["popularity_score"]), 4),
                    "has_interacted": False,
                    "rationale": f"Foundational curriculum item: highly accessed {act_type} recommended for new learners.",
                }
            )
        return results

    def _generate_rationale(self, activity_type: str, risk_level: str, has_interacted: bool) -> str:
        if risk_level in ["high", "critical"]:
            if activity_type == "quiz":
                return "Targeted formative quiz recommended to reinforce core competencies and diagnose knowledge gaps."
            elif activity_type == "oucontent":
                return "Essential primary learning module to establish concept mastery and prepare for upcoming assessments."
            elif activity_type == "forumng":
                return "Course peer discussion forum to seek clarification on challenging course topics."
            else:
                return f"Curated {activity_type} resource aligned with student study requirements."
        else:
            if activity_type == "forumng":
                return "Active peer discussion space to collaborate on advanced synthesis and problem-solving."
            elif activity_type == "quiz":
                return "Practice assessment to validate mastery and benchmark exam readiness."
            else:
                return f"High-yield {activity_type} resource correlated with cohort academic distinction."


if __name__ == "__main__":
    engine = HybridRecommendationEngine()
    recs = engine.recommend_for_student(student_id=100001, top_n=5, risk_level="high")
    print("Sample recommendations:")
    for r in recs:
        print(
            f" - ID: {r['resource_id']} | Type: {r['activity_type']} | Score: {r['affinity_score']} | Reason: {r['rationale']}"
        )
