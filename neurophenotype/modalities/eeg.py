"""
EEG Modality — Neural Signatures
Hardware: OpenBCI Cyton or Muse headset
Features: Band power (delta, theta, alpha, beta, gamma), spike detection,
          inter-electrode coherence, phase-amplitude coupling
"""
import numpy as np
from .base import BaseModality


class EEGModality(BaseModality):
    # Feature vector length: 5 band powers + coherence + spike rate + PAC = 8
    FEATURE_DIM = 8

    def collect(self) -> any:
        # TODO: Connect to OpenBCI/Muse stream (brainflow or pylsl)
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: Bandpass filter, artifact rejection, epoching
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: Compute delta/theta/alpha/beta/gamma power, coherence, spike rate, PAC
        raise NotImplementedError
