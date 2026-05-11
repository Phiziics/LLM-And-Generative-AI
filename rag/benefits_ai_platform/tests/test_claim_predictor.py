from pathlib import Path

from src.models.predict_claim_risk import ClaimRiskPredictor


def test_claim_predictor_loads_model():
    project_root = Path.cwd()
    predictor = ClaimRiskPredictor(project_root=project_root)

    assert predictor.model is not None


def test_claim_predictor_returns_expected_keys():
    project_root = Path.cwd()
    predictor = ClaimRiskPredictor(project_root=project_root)

    result = predictor.predict_one(
        claim_type="inpatient",
        claim_duration_days=5,
        has_provider_id=1,
        has_diagnosis_code=1,
    )

    assert "predicted_high_cost_claim" in result
    assert "high_cost_claim_probability" in result
    assert "human_review_required" in result

    assert result["predicted_high_cost_claim"] in [0, 1]
    assert 0 <= result["high_cost_claim_probability"] <= 1
    assert isinstance(result["human_review_required"], bool)
    