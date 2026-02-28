"""
XGBoost classifier wrapper — train and predict disorder probabilities.
Output: probability score per target condition (Rett, Dravet, Angelman)
"""
import numpy as np
import xgboost as xgb
import joblib


CONDITION_LABELS = ["rett_syndrome", "dravet_syndrome", "angelman_syndrome"]

GENOMIC_PANELS = {
    "rett_syndrome": {
        "panel": "MECP2 sequencing + deletion/duplication analysis",
        "cost": 400,
        "wes_cost": 5000,
    },
    "dravet_syndrome": {
        "panel": "SCN1A sequencing",
        "cost": 300,
        "wes_cost": 4800,
    },
    "angelman_syndrome": {
        "panel": "Chromosome 15 methylation analysis + UBE3A sequencing",
        "cost": 500,
        "wes_cost": 4800,
    },
}


class NeuroPhenotypeClassifier:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None

    def predict(self, feature_vector: np.ndarray) -> dict:
        """Return probability scores and recommended genomic panels."""
        if self.model is None:
            raise RuntimeError("No model loaded. Run classifier/train.py first.")

        x = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]

        results = {}
        for label, prob in zip(CONDITION_LABELS, probs):
            panel_info = GENOMIC_PANELS[label]
            results[label] = {
                "probability": round(float(prob), 4),
                "recommended_panel": panel_info["panel"],
                "estimated_cost": panel_info["cost"],
                "estimated_savings": panel_info["wes_cost"] - panel_info["cost"],
            }

        return dict(sorted(results.items(), key=lambda x: x[1]["probability"], reverse=True))

    def save(self, path: str):
        joblib.dump(self.model, path)
