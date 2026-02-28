"""
rPPG Modality — Cardiovascular via Webcam
Hardware: Laptop webcam
Library: OpenCV-based rPPG (pyVHR or similar)
Features: Heart rate, respiratory rate, pulse waveform morphology
"""
import numpy as np
from .base import BaseModality


class RPPGModality(BaseModality):
    # Feature vector: HR, RR, pulse amplitude, pulse regularity = 4
    FEATURE_DIM = 4

    def collect(self) -> any:
        # TODO: Capture frames from webcam via cv2.VideoCapture
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: ROI detection (face/forehead), color channel extraction, detrend
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: FFT-based HR estimation, respiratory rate from signal envelope
        raise NotImplementedError
