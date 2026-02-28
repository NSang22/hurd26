"""
GSR Modality — Galvanic Skin Response (Sympathetic Reactivity)
Hardware: Arduino Nano 33 BLE Sense + custom GSR circuit
          (two finger electrodes → 100kΩ voltage divider → analog A0)
Features: Skin conductance level (SCL), SCR peaks, sympathetic reactivity
"""
import numpy as np
from .base import BaseModality


class GSRModality(BaseModality):
    # Feature vector: SCL mean, SCL std, SCR count, SCR amplitude mean = 4
    FEATURE_DIM = 4

    def collect(self) -> any:
        # TODO: Stream analog A0 from Arduino via BLE or serial
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: Low-pass filter (cutoff ~5 Hz) to remove motion artifacts
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: Decompose into tonic (SCL) + phasic (SCR) components
        raise NotImplementedError
