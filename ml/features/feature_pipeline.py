"""
Feature Engineering and Dataset Pipeline.
Extracts student-level demographic, behavioral, and assessment features with temporal leakage prevention.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class FeaturePipeline:
    def __init__(
        self, raw_data_dir: Path = Path("data/raw"), processed_dir: Path = Path("data/processed")
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        self.df_student_info = pd.read_csv(self.raw_data_dir / "studentInfo.csv")
        self.df_student_reg = pd.read_csv(self.raw_data_dir / "studentRegistration.csv")
        self.df_student_assess = pd.read_csv(self.raw_data_dir / "studentAssessment.csv")
        self.df_assessments = pd.read_csv(self.raw_data_dir / "assessments.csv")
        self.df_student_vle = pd.read_csv(self.raw_data_dir / "studentVle.csv")
        self.df_vle = pd.read_csv(self.raw_data_dir / "vle.csv")
        self.df_courses = pd.read_csv(self.raw_data_dir / "courses.csv")

    def engineer_vle_features(self, cutoff_day: int = 100) -> pd.DataFrame:
        """
        Calculates engagement features up to an early-warning cutoff day (e.g. day 100)
        to prevent future data leakage for early risk detection.
        """
        # Filter clicks up to cutoff_day
        vle_early = self.df_student_vle[self.df_student_vle["date"] <= cutoff_day].copy()

        # Merge with VLE resource activity types
        vle_merged = vle_early.merge(
            self.df_vle[["id_site", "activity_type"]], on="id_site", how="left"
        )

        # General aggregation
        grp = vle_merged.groupby(["code_module", "code_presentation", "id_student"])
        agg_general = grp.agg(
            total_clicks=("sum_click", "sum"),
            unique_resources=("id_site", "nunique"),
            active_days=("date", "nunique"),
            early_click_mean=("sum_click", "mean"),
        ).reset_index()

        agg_general["clicks_per_active_day"] = (
            agg_general["total_clicks"] / agg_general["active_days"].clip(lower=1)
        ).round(2)

        # Pivot by activity type for top categories
        top_types = ["forumng", "oucontent", "quiz", "resource", "url", "subpage"]
        vle_top = vle_merged[vle_merged["activity_type"].isin(top_types)]
        pivot_types = vle_top.pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        # Rename columns
        rename_cols = {t: f"clicks_{t}" for t in top_types if t in pivot_types.columns}
        pivot_types = pivot_types.rename(columns=rename_cols)

        # Merge general and pivoted features
        vle_features = agg_general.merge(
            pivot_types, on=["code_module", "code_presentation", "id_student"], how="left"
        ).fillna(0)

        return vle_features

    def engineer_assessment_features(self, cutoff_day: int = 100) -> pd.DataFrame:
        """
        Calculates assessment performance metrics up to the early cutoff day.
        """
        assess_merged = self.df_student_assess.merge(
            self.df_assessments[
                [
                    "id_assessment",
                    "code_module",
                    "code_presentation",
                    "date",
                    "weight",
                    "assessment_type",
                ]
            ],
            on="id_assessment",
            how="inner",
        )
        # Keep early assessments
        assess_early = assess_merged[assess_merged["date"] <= cutoff_day].copy()

        # Calculate submission delay relative to due date
        assess_early["submission_delay"] = assess_early["date_submitted"] - assess_early["date"]
        assess_early["is_late"] = (assess_early["submission_delay"] > 0).astype(int)

        grp = assess_early.groupby(["code_module", "code_presentation", "id_student"])
        assess_features = (
            grp.agg(
                assessments_taken=("id_assessment", "count"),
                score_mean=("score", "mean"),
                score_min=("score", "min"),
                score_max=("score", "max"),
                score_std=("score", lambda x: x.std() if len(x) > 1 else 0.0),
                late_submissions_count=("is_late", "sum"),
                mean_delay_days=("submission_delay", "mean"),
            )
            .reset_index()
            .fillna(0)
        )

        return assess_features

    def build_dataset(self, cutoff_day: int = 100) -> pd.DataFrame:
        self.load_raw_data()

        # 1. Base student records
        df = self.df_student_info.copy()

        # 2. Merge registration info
        df = df.merge(
            self.df_student_reg[
                ["code_module", "code_presentation", "id_student", "date_registration"]
            ],
            on=["code_module", "code_presentation", "id_student"],
            how="left",
        )
        df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

        # 3. Merge VLE engagement features
        vle_feat = self.engineer_vle_features(cutoff_day=cutoff_day)
        df = df.merge(vle_feat, on=["code_module", "code_presentation", "id_student"], how="left")

        # 4. Merge assessment features
        assess_feat = self.engineer_assessment_features(cutoff_day=cutoff_day)
        df = df.merge(
            assess_feat, on=["code_module", "code_presentation", "id_student"], how="left"
        )

        # Fill missing values for students without engagement yet
        numeric_fill_zeros = [
            "total_clicks",
            "unique_resources",
            "active_days",
            "clicks_per_active_day",
            "early_click_mean",
            "assessments_taken",
            "score_mean",
            "score_min",
            "score_max",
            "score_std",
            "late_submissions_count",
            "mean_delay_days",
        ]
        for col in numeric_fill_zeros:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for col in df.columns:
            if col.startswith("clicks_"):
                df[col] = df[col].fillna(0)

        # 5. Define Target Labels
        # Academic Outcome (4 classes: Distinction, Pass, Fail, Withdrawn)
        df["target_outcome"] = df["final_result"]
        # Early Academic Risk: 1 = At-Risk (Fail or Withdrawn), 0 = Normal/High (Pass or Distinction)
        df["target_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
        # Academic Success: 1 = Pass/Distinction, 0 = Fail/Withdrawn
        df["target_passed"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)

        # 6. Encode Categorical Demographics
        df["gender_code"] = (df["gender"] == "M").astype(int)
        df["disability_code"] = (df["disability"] == "Y").astype(int)

        age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
        df["age_band_code"] = df["age_band"].map(age_map).fillna(0).astype(int)

        edu_map = {
            "No Formal quals": 0,
            "Lower Than A Level": 1,
            "A Level or Equivalent": 2,
            "HE Qualification": 3,
            "Post Graduate Qualification": 4,
        }
        df["education_level_code"] = df["highest_education"].map(edu_map).fillna(1).astype(int)

        # Imd band (ordinal ranking: 0 to 9)
        imd_order = [
            "0-10%",
            "10-20",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
        ]
        imd_map = {band: i for i, band in enumerate(imd_order)}
        df["imd_band_code"] = df["imd_band"].map(imd_map).fillna(4).astype(int)

        return df

    def save_processed_splits(self, test_size: float = 0.20, random_state: int = 42):
        df = self.build_dataset()

        # Save full feature dataset
        df.to_csv(self.processed_dir / "features_full.csv", index=False)

        # Stratified train/test split on target_risk
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["target_risk"]
        )

        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)

        metadata = {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "risk_prevalence_train": float(train_df["target_risk"].mean()),
            "risk_prevalence_test": float(test_df["target_risk"].mean()),
            "feature_columns": [
                c
                for c in df.columns
                if c
                not in [
                    "final_result",
                    "target_outcome",
                    "target_risk",
                    "target_passed",
                    "code_module",
                    "code_presentation",
                    "id_student",
                    "gender",
                    "region",
                    "highest_education",
                    "imd_band",
                    "age_band",
                    "disability",
                ]
            ],
        }
        with open(self.processed_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Saved processed datasets to {self.processed_dir}. Train: {len(train_df)}, Test: {len(test_df)}"
        )
        return train_df, test_df, metadata


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.save_processed_splits()
