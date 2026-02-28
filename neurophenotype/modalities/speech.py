"""
Speech Modality — Vocal Biomarkers
Hardware: Laptop microphone
Libraries: librosa (pitch, rhythm, pause patterns, formants),
           ElevenLabs API (optional transcription/prosody)
Features: Pitch variability, speech rhythm, pause duration,
          phoneme clarity, vocalization rate
"""
import numpy as np
from .base import BaseModality


class SpeechModality(BaseModality):
    # Feature vector: pitch mean, pitch std, speech rhythm, mean pause duration,
    #                 pause rate, vocalization rate = 6
    FEATURE_DIM = 6

    def collect(self) -> any:
        # TODO: Record audio via sounddevice or pyaudio
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: VAD (voice activity detection), noise reduction
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: librosa pitch tracking, rhythm estimation, pause segmentation,
        #       MFCCs for phoneme clarity proxy
        raise NotImplementedError
