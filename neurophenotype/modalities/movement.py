"""
Movement Modality — Motor Phenotyping
Hardware: Laptop webcam (MediaPipe) + Arduino Nano 33 BLE Sense IMU (wrist-worn)
Libraries: MediaPipe Pose/Hands, Arduino IMU streaming via BLE
Features: Hand stereotypy detection (midline repetitive movements), tremor frequency,
          movement irregularity, upper limb rhythm
"""
import numpy as np
from .base import BaseModality


class MovementModality(BaseModality):
    # Feature vector: stereotypy score, tremor freq, movement irregularity,
    #                 limb rhythm, IMU accel std x/y/z = 7
    FEATURE_DIM = 7

    def collect(self) -> any:
        # TODO: Simultaneously capture MediaPipe landmarks + Arduino BLE IMU stream
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: Smooth landmark trajectories, calibrate IMU axes
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: Detect midline hand crossings (Rett stereotypy), FFT of wrist IMU
        #       for tremor frequency, compute movement irregularity (sample entropy)
        raise NotImplementedError
