from abc import ABC, abstractmethod
import numpy as np


class BaseModality(ABC):
    @abstractmethod
    def collect(self) -> any:
        """Collect raw data from hardware/software source"""
        pass

    @abstractmethod
    def preprocess(self, raw_data: any) -> any:
        """Clean, filter, epoch raw data"""
        pass

    @abstractmethod
    def extract_features(self, processed_data: any) -> np.ndarray:
        """Return fixed-length 1D feature vector"""
        pass

    def run(self) -> np.ndarray:
        raw = self.collect()
        processed = self.preprocess(raw)
        return self.extract_features(processed)
