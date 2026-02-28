"""
Keyboard & Mouse Modality — Fine Motor
Hardware: Standard keyboard and mouse (passively logged)
Features: Key hold time, inter-keystroke interval (IKI), mouse velocity,
          positional accuracy, error rate
"""
import numpy as np
from .base import BaseModality


class KeyboardMouseModality(BaseModality):
    # Feature vector: mean hold time, hold time std, mean IKI, IKI std,
    #                 mouse velocity mean, mouse velocity std, error rate = 7
    FEATURE_DIM = 7

    def collect(self) -> any:
        # TODO: Use pynput to passively log keyboard/mouse events
        raise NotImplementedError

    def preprocess(self, raw_data: any) -> any:
        # TODO: Filter outliers, align timestamps
        raise NotImplementedError

    def extract_features(self, processed_data: any) -> np.ndarray:
        # TODO: Compute hold times, IKIs, mouse speed/accuracy statistics
        raise NotImplementedError
