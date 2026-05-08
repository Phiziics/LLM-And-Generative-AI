from pathlib import Path

import joblib
import pandas as pd


class ClaimRiskPredictor:
    def __init__(
        self,
        project_root: str | Path | None = None,
        model_path: str | Path | None = None,
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()

        self.model_path = (
            Path(model_path)
            if model_path
            else self.project_root / "models" / "claim_risk_model.pkl"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.feature_columns = [
            "claim_type",
            "claim_duration_days",
            "has_provider_id",
            "has_diagnosis_code",
        ]

    def predict_one(
        self,
        claim_type: str,
        claim_duration_days: float,
        has_provider_id: int,
        has_diagnosis_code: int,
    ) -> dict:
        input_df = pd.DataFrame(
            [
                {
                    "claim_type": claim_type,
                    "claim_duration_days": claim_duration_days,
                    "has_provider_id": has_provider_id,
                    "has_diagnosis_code": has_diagnosis_code,
                }
            ]
        )

        prediction = int(self.model.predict(input_df)[0])
        probability = float(self.model.predict_proba(input_df)[0, 1])

        return {
            "predicted_high_cost_claim": prediction,
            "high_cost_claim_probability": probability,
            "human_review_required": probability >= 0.50,
        }

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [
            col for col in self.feature_columns
            if col not in input_df.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        prediction_df = input_df.copy()

        prediction_df["predicted_high_cost_claim"] = self.model.predict(
            prediction_df[self.feature_columns]
        )

        prediction_df["high_cost_claim_probability"] = self.model.predict_proba(
            prediction_df[self.feature_columns]
        )[:, 1]

        prediction_df["human_review_required"] = (
            prediction_df["high_cost_claim_probability"] >= 0.50
        )

        return prediction_df


if __name__ == "__main__":
    predictor = ClaimRiskPredictor()

    result = predictor.predict_one(
        claim_type="inpatient",
        claim_duration_days=5,
        has_provider_id=1,
        has_diagnosis_code=1,
    )

    print(result)