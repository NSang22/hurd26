import inspect
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

# The current fused classifier is trained only on these live biosignal groups.
SUPPORTED_LIVE_MODALITIES = ("eeg", "hrv", "movement", "speech")


def get_feature_vector(
    config_path="config.yaml",
    mock_profile: str = None,
    movement_task: str = None,
    speech_task: str = None,
) -> np.ndarray:
    """
    Collect and concatenate feature vectors from all active supported modalities.

    Args:
        config_path:    Path to config.yaml.
        mock_profile:   If set, passes mock_profile to modalities that support it
                        instead of hitting real hardware.
                        Options: "rett", "dravet", "angelman".
        movement_task:  Guided task name forwarded to MovementModality.
        speech_task:    Guided task name forwarded to SpeechModality.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vectors = []
    enabled = config.get("modalities", {})

    unsupported_enabled = [
        name for name, active in enabled.items()
        if active and name not in SUPPORTED_LIVE_MODALITIES
    ]
    if unsupported_enabled:
        print(
            "[Fusion] Ignoring unsupported live modalities for current fused model: "
            + ", ".join(unsupported_enabled)
        )

    for name in SUPPORTED_LIVE_MODALITIES:
        if not enabled.get(name, False):
            continue

        cls = MODALITY_MAP[name]
        try:
            sig = inspect.signature(cls.__init__).parameters
            kwargs: dict = {}
            if mock_profile and "mock_profile" in sig:
                kwargs["mock_profile"] = mock_profile
            if name == "movement" and movement_task and "task" in sig:
                kwargs["task"] = movement_task
            if name == "speech" and speech_task and "task" in sig:
                kwargs["task"] = speech_task
            modality = cls(**kwargs)

            vec = modality.run()
            vectors.append(np.array(vec, dtype=np.float32).flatten())

        except Exception as e:
            print(f"[Fusion] Error in '{name}': {e} — using zeros")
            vectors.append(np.zeros(getattr(cls, "FEATURE_DIM", 4), dtype=np.float32))

    if not vectors:
        raise RuntimeError(
            "No supported live modalities are enabled. "
            f"Enable at least one of: {', '.join(SUPPORTED_LIVE_MODALITIES)}"
        )

    return np.concatenate(vectors)