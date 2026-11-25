"""
Unit tests for Machine Learning models and inference validation.
"""

import json
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path("artifacts/models")
DATA_DIR = Path("data/processed")


def test_saved_models_exist():
    """Verify that all champion and candidate model artifacts exist."""
    expected_models = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "gradient_boosting.joblib",
        "xgboost.joblib",
        "champion_risk_model.joblib",
        "scaler.joblib",
        "tabular_mlp.pt",
        "behavior_gru.pt",
        "graph_embeddings.npz",
    ]
    for model_file in expected_models:
        path = MODELS_DIR / model_file
        assert path.exists(), f"Missing model artifact: {path}"


def test_model_inference_and_calibration():
    """Verify champion model can perform inference and outputs valid probabilities."""
    gb_path = MODELS_DIR / "champion_risk_model.joblib"
    meta_path = DATA_DIR / "dataset_metadata.json"

    assert meta_path.exists()
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    model = joblib.load(gb_path)
    feature_cols = metadata["feature_columns"]

    # Generate synthetic input row
    dummy_input = np.random.uniform(10.0, 80.0, size=(5, len(feature_cols)))

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (5, 2)
    # Probabilities must sum to 1.0
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)
    # Values between 0 and 1
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_risk_tier_classification():
    """Verify risk probabilities map to reasonable risk categories."""
    def assign_tier(prob: float) -> str:
        if prob >= 0.70:
            return "Critical"
        elif prob >= 0.35:
            return "Elevated"
        return "Low"

    assert assign_tier(0.15) == "Low"
    assert assign_tier(0.50) == "Elevated"
    assert assign_tier(0.85) == "Critical"
