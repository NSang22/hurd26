import yaml
import numpy as np
from modalities import eeg, hrv, gsr, rppg, movement, speech, keyboard_mouse

MODALITY_MAP = {
    "eeg": eeg.EEGModality,
    "hrv": hrv.HRVModality,
    "gsr": gsr.GSRModality,
    "rppg": rppg.RPPGModality,
    "movement": movement.MovementModality,
    "speech": speech.SpeechModality,
    "keyboard_mouse": keyboard_mouse.KeyboardMouseModality,
}


def get_feature_vector(config_path="config.yaml") -> np.ndarray:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vectors = []
    for name, active in config["modalities"].items():
        if active:
            modality = MODALITY_MAP[name]()
            vectors.append(modality.run())

    return np.concatenate(vectors)
