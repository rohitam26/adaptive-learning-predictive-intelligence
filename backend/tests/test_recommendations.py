"""
Unit tests for Hybrid Recommendation Engine.
Tests collaborative filtering, graph embeddings, engagement scoring, and cold-start fallbacks.
"""

import pytest

from ml.recommendation.hybrid_engine import HybridRecommendationEngine


@pytest.fixture(scope="module")
def engine():
    return HybridRecommendationEngine()


def test_recommendations_known_student(engine):
    """Verify recommendations for an enrolled student."""
    student_id = 100001
    recs = engine.recommend_for_student(student_id=student_id, top_n=5)
    
    assert len(recs) <= 5
    assert len(recs) > 0
    
    first = recs[0]
    assert "resource_id" in first
    assert "activity_type" in first
    assert "affinity_score" in first
    assert "rationale" in first
    assert 0.0 <= first["affinity_score"] <= 2.5


def test_recommendations_cold_start_fallback(engine):
    """Verify recommendation gracefully handles unknown/cold-start student IDs."""
    unknown_student_id = 99999999
    recs = engine.recommend_for_student(student_id=unknown_student_id, top_n=5)

    assert len(recs) > 0
    # Must provide popular fallbacks with transparent rationale
    assert any("curriculum" in r.get("rationale", "").lower() or "recommended" in r.get("rationale", "").lower() for r in recs)
