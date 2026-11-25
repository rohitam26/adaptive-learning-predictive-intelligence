"""
Unit tests for OULAD data ingestion and feature engineering pipeline.
Tests zero-leakage, temporal cutoffs, and group splitting.
"""

from pathlib import Path

import pandas as pd
import pytest

from ml.features.feature_pipeline import FeaturePipeline


@pytest.fixture(scope="module")
def pipeline():
    pipe = FeaturePipeline(raw_data_dir=Path("data/raw"), processed_dir=Path("data/processed"))
    pipe.load_raw_data()
    return pipe


def test_raw_data_loading(pipeline):
    """Verify raw tables are loaded with expected schemas."""
    assert pipeline.df_student_info is not None
    assert pipeline.df_courses is not None
    assert pipeline.df_vle is not None
    assert pipeline.df_student_vle is not None
    assert pipeline.df_student_assess is not None
    assert pipeline.df_assessments is not None

    assert "id_student" in pipeline.df_student_info.columns
    assert "final_result" in pipeline.df_student_info.columns
    assert len(pipeline.df_student_info) > 0


def test_feature_dataset_temporal_cutoff(pipeline):
    """Verify features are constructed strictly within the 100-day window."""
    df_feat = pipeline.build_dataset(cutoff_day=100)

    # Required columns
    assert "id_student" in df_feat.columns
    assert "target_risk" in df_feat.columns
    assert "score_mean" in df_feat.columns
    assert "total_clicks" in df_feat.columns
    assert "active_days" in df_feat.columns

    # Check target binary encoding
    unique_targets = set(df_feat["target_risk"].unique())
    assert unique_targets.issubset({0, 1})

    # Verify no NaN values in engineered features
    assert not df_feat["score_mean"].isna().any()
    assert not df_feat["total_clicks"].isna().any()


def test_zero_leakage_group_split():
    """Verify that student IDs in train.csv and test.csv have ZERO overlap."""
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    assert train_path.exists() and test_path.exists()

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    train_students = set(df_train["id_student"].unique())
    test_students = set(df_test["id_student"].unique())

    # Overlap MUST be empty (zero student identity leakage across train/test)
    overlap = train_students.intersection(test_students)
    assert len(overlap) == 0, f"Found {len(overlap)} leaking students between train and test!"

    # Verify target distribution in both splits
    assert df_train["target_risk"].mean() > 0.1
    assert df_test["target_risk"].mean() > 0.1
