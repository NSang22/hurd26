"""
HRV Modality — Autonomic Nervous System
Hardware: Apple Watch via HealthKit API
Features: SDNN, RMSSD, pNN50, LF/HF ratio (sympathovagal balance)
"""
import numpy as np
from .base import BaseModality


class HRVModality(BaseModality):
    # Feature vector: SDNN, RMSSD, pNN50, LF/HF = 4
    FEATURE_DIM = 4

    def collect(self) -> any:
        # TODO: Pull HRV data from HealthKit export / Apple Health CSV
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: Extract RR intervals, filter ectopic beats
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: Compute SDNN, RMSSD, pNN50, LF/HF via FFT of RR intervals
        raise NotImplementedError
