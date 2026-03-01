"""
XGBoost classifier wrapper for fused NeuroPhenotype inference.

Supports:
- legacy biosignal-only models
- fused biosignal + clinical models
- automatic padding/truncation to the trained feature width
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np


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
    "control": {
        "panel": "No targeted panel indicated",
        "cost": 0,
        "wes_cost": 0,
    },
}


class NeuroPhenotypeClassifier:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.label_encoder = None
        self.classes_: list[str] = []
        self.feature_names: list[str] = []
        self.feature_groups: dict[str, int] = {}
        self.feature_slices: dict[str, tuple[int, int]] = {}
        self.expected_n_features = 0

        if model_path:
            data = joblib.load(model_path)
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]
            self.classes_ = list(self.label_encoder.classes_)
            self.feature_names = list(data.get("feature_names", []))
            self.feature_groups = dict(data.get("feature_groups", {}))
            self.feature_slices = dict(data.get("feature_slices", {}))

            if self.feature_names:
                self.expected_n_features = len(self.feature_names)
            elif hasattr(self.model, "n_features_in_"):
                self.expected_n_features = int(self.model.n_features_in_)

    def build_fused_feature_vector(
        self,
        feature_vector: np.ndarray | list[float],
        clinical_vector: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        """
        Build a 1D feature vector aligned to the trained model width.

        If `clinical_vector` is supplied, it is appended to the biosignal vector.
        The combined vector is then padded or truncated to the trained width.
        """
        biosignal = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        if clinical_vector is None:
            fused = biosignal
        else:
            clinical = np.asarray(clinical_vector, dtype=np.float32).reshape(-1)
            fused = np.concatenate([biosignal, clinical]).astype(np.float32)

        if self.expected_n_features <= 0:
            return fused

        if fused.size < self.expected_n_features:
            padding = np.zeros(self.expected_n_features - fused.size, dtype=np.float32)
            fused = np.concatenate([fused, padding]).astype(np.float32)
        elif fused.size > self.expected_n_features:
            fused = fused[: self.expected_n_features].astype(np.float32)

        return fused

    def predict(
        self,
        feature_vector: np.ndarray | list[float],
        clinical_vector: np.ndarray | list[float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("No model loaded. Run classifier/train.py first.")

        fused = self.build_fused_feature_vector(feature_vector, clinical_vector=clinical_vector)
        x = fused.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]

        results: dict[str, dict[str, Any]] = {}
        for label, prob in zip(self.classes_, probs):
            panel_info = GENOMIC_PANELS.get(label, {"panel": "Unknown", "cost": 0, "wes_cost": 0})
            results[label] = {
                "probability": round(float(prob), 4),
                "recommended_panel": panel_info["panel"],
                "estimated_cost": panel_info["cost"],
                "estimated_savings": panel_info["wes_cost"] - panel_info["cost"],
            }

        return dict(sorted(results.items(), key=lambda item: item[1]["probability"], reverse=True))

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "feature_groups": self.feature_groups,
                "feature_slices": self.feature_slices,
            },
            path,
        )
