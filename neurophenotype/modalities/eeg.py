"""
EEG Modality — Neural Signatures
Hardware: OpenBCI Cyton or Muse headset (live), or EDF file (offline/demo)
Features: Band power (delta, theta, alpha, beta, gamma), spike detection,
          inter-electrode coherence, phase-amplitude coupling

FEATURE VECTOR (length 23):
  [0]    delta power (1–4 Hz)      ← primary: Angelman, Rett
  [1]    theta power (4–8 Hz)      ← primary: Rett background slowing
  [2]    alpha power (8–13 Hz)
  [3]    beta  power (13–30 Hz)
  [4]    gamma power (30–45 Hz)
  [5]    delta/theta ratio          ← Angelman vs Rett discrimination
  [6]    theta/alpha ratio          ← Rett severity proxy
  [7]    spike_rate (spikes/min)   ← Dravet: spike-wave at 2–3.5 Hz
  [8]    spike_amplitude_mean
  [9]    spike_dominance_freq       ← distinguishes 2–3.5 Hz Dravet vs 1–2 Hz other
  [10]   mean_coherence             ← global synchrony
  [11]   frontal_coherence          ← Angelman frontal notching signature
  [12]   PAC (theta→gamma)          ← cortical coupling; reduced in MECP2/SCN1A
  [13]   background_slowing_index   ← Rett: shift of spectral mass to low freqs
  [14]   frontal_delta_asymmetry    ← lateralization marker
  [15]   inter_burst_interval       ← Angelman burst-suppression-like pattern
  [16–22] per-channel delta power (up to 7 channels, zero-padded)

Usage:
  # Offline (demo / classifier training):
  m = EEGModality(edf_path="data/public/chbmit/chb01/chb01_01.edf")
  features = m.run()

  # Live stream (OpenBCI / Muse via BrainFlow):
  m = EEGModality(board_id=0, serial_port="COM3")   # BrainFlow board_id
  features = m.run()
"""

import numpy as np
from pathlib import Path
from .base import BaseModality

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully so other modalities still work
# ---------------------------------------------------------------------------
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_DIM = 23
SFREQ_TARGET = 256          # Hz — resample target
EPOCH_DURATION = 4.0        # seconds per epoch
OVERLAP = 0.5               # 50% overlap
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SPIKE_THRESHOLD_ZSCORE = 3.5   # z-score threshold for spike detection
MAX_CHANNELS = 7               # per-channel features (zero-pad if fewer)


class EEGModality(BaseModality):
    FEATURE_DIM = FEATURE_DIM

    def __init__(
        self,
        edf_path: str = None,
        board_id: int = None,
        serial_port: str = None,
        epoch_s: float = EPOCH_DURATION,
        n_epochs: int = 10,
    ):
        """
        Args:
            edf_path:    Path to .edf file (offline mode).
            board_id:    BrainFlow board id (live mode). E.g. 0 = Cyton, 2 = Muse.
            serial_port: Serial port for BrainFlow (e.g. "COM3" or "/dev/ttyUSB0").
            epoch_s:     Epoch duration in seconds.
            n_epochs:    How many epochs to collect in live mode.
        """
        self.edf_path = edf_path
        self.board_id = board_id
        self.serial_port = serial_port
        self.epoch_s = epoch_s
        self.n_epochs = n_epochs

    # ------------------------------------------------------------------
    # BaseModality interface
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        """Return dict with keys 'data' (n_ch x n_samples) and 'sfreq'."""
        if self.edf_path is not None:
            return self._collect_from_edf(self.edf_path)
        elif self.board_id is not None:
            return self._collect_live()
        else:
            raise ValueError("Provide either edf_path (offline) or board_id (live).")

    def preprocess(self, raw: dict) -> dict:
        """Bandpass filter, notch filter, resample, simple artifact rejection."""
        data = raw["data"].astype(np.float64)
        sfreq = raw["sfreq"]

        if MNE_AVAILABLE:
            data, sfreq = self._preprocess_mne(data, sfreq)
        else:
            data, sfreq = self._preprocess_numpy(data, sfreq)

        return {"data": data, "sfreq": sfreq}

    def extract_features(self, processed: dict) -> np.ndarray:
        data = processed["data"]
        sfreq = processed["sfreq"]

        epochs = self._epoch(data, sfreq)          # (n_epochs, n_ch, n_samples)
        if len(epochs) == 0:
            return np.zeros(FEATURE_DIM)

        band_powers = self._band_powers(epochs, sfreq)          # (n_ep, 5)
        spike_feats = self._spike_features(epochs, sfreq)       # (n_ep, 3)
        coherence   = self._coherence_features(epochs, sfreq)   # (n_ep, 2)
        pac         = self._pac(epochs, sfreq)                   # (n_ep,)
        extra       = self._extra_features(epochs, sfreq, band_powers)  # (n_ep, 3)
        per_ch_delta = self._per_channel_delta(epochs, sfreq)   # (n_ep, MAX_CH)

        # Average over epochs
        bp   = band_powers.mean(axis=0)      # (5,)
        sp   = spike_feats.mean(axis=0)      # (3,)
        coh  = coherence.mean(axis=0)        # (2,)
        pac_ = float(pac.mean())             # scalar
        ex   = extra.mean(axis=0)            # (3,)
        pcd  = per_ch_delta.mean(axis=0)     # (MAX_CH,)

        # Ratio features
        d_t_ratio = bp[0] / (bp[1] + 1e-8)
        t_a_ratio = bp[1] / (bp[2] + 1e-8)

        vec = np.concatenate([
            bp,                          # [0–4]   band powers
            [d_t_ratio, t_a_ratio],      # [5–6]   ratios
            sp,                          # [7–9]   spike features
            coh,                         # [10–11] coherence
            [pac_],                      # [12]    PAC
            ex,                          # [13–15] extra
            pcd,                         # [16–22] per-channel delta
        ])
        assert vec.shape[0] == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {vec.shape[0]}"
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def _collect_from_edf(self, path: str) -> dict:
        if not MNE_AVAILABLE:
            raise ImportError("pip install mne to load EDF files.")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()          # (n_ch, n_samples)
        sfreq = raw.info["sfreq"]
        return {"data": data, "sfreq": sfreq}

    def _collect_live(self) -> dict:
        if not BRAINFLOW_AVAILABLE:
            raise ImportError("pip install brainflow for live acquisition.")

        params = BrainFlowInputParams()
        if self.serial_port:
            params.serial_port = self.serial_port

        board = BoardShim(self.board_id, params)
        board.prepare_session()
        board.start_stream()

        import time
        time.sleep(self.epoch_s * self.n_epochs + 2.0)

        raw = board.get_board_data()
        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        data = raw[eeg_channels, :]
        sfreq = BoardShim.get_sampling_rate(self.board_id)
        return {"data": data, "sfreq": sfreq}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_mne(self, data: np.ndarray, sfreq: float):
        info = mne.create_info(
            ch_names=[f"ch{i}" for i in range(data.shape[0])],
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        # Notch filter (50 Hz and 60 Hz power line)
        raw.notch_filter([50.0, 60.0], verbose=False)

        # Bandpass 0.5–45 Hz — keep delta!
        raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

        # Resample
        if sfreq != SFREQ_TARGET:
            raw.resample(SFREQ_TARGET, verbose=False)

        return raw.get_data(), SFREQ_TARGET

    def _preprocess_numpy(self, data: np.ndarray, sfreq: float):
        """Fallback when MNE is not installed — basic numpy filtering."""
        from scipy.signal import butter, filtfilt, iirnotch

        def bandpass(x, lo, hi, fs):
            b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
            return filtfilt(b, a, x, axis=-1)

        def notch(x, freq, fs):
            b, a = iirnotch(freq / (fs / 2), 30.0)
            return filtfilt(b, a, x, axis=-1)

        data = notch(data, 50.0, sfreq)
        data = notch(data, 60.0, sfreq)
        data = bandpass(data, 0.5, 45.0, sfreq)
        return data, sfreq

    # ------------------------------------------------------------------
    # Epoching
    # ------------------------------------------------------------------

    def _epoch(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Split data into overlapping epochs. Returns (n_ep, n_ch, n_samp)."""
        n_samp = int(self.epoch_s * sfreq)
        step   = int(n_samp * (1 - OVERLAP))
        n_ch, total = data.shape
        epochs = []
        start = 0
        while start + n_samp <= total:
            epoch = data[:, start:start + n_samp]
            # Simple amplitude-based artifact rejection
            if np.abs(epoch).max() < 500e-6:   # 500 µV threshold
                epochs.append(epoch)
            start += step
        if len(epochs) == 0:
            return np.zeros((1, n_ch, n_samp))
        return np.stack(epochs, axis=0)   # (n_ep, n_ch, n_samp)

    # ------------------------------------------------------------------
    # Feature extractors (all return shape (n_ep, ...))
    # ------------------------------------------------------------------

    def _band_powers(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """Compute log band power per epoch averaged across channels."""
        from scipy.signal import welch
        n_ep, n_ch, n_samp = epochs.shape
        nperseg = min(n_samp, int(sfreq * 2))
        out = np.zeros((n_ep, 5))
        band_list = list(FREQ_BANDS.values())

        for i, epoch in enumerate(epochs):
            # Average PSD across channels
            freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
            psd_mean = psd.mean(axis=0)   # (n_freqs,)
            for j, (lo, hi) in enumerate(band_list):
                mask = (freqs >= lo) & (freqs < hi)
                out[i, j] = np.log1p(psd_mean[mask].mean())
        return out

    def _spike_features(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Detect sharp spikes (proxy for spike-wave discharges).
        Returns [spike_rate, mean_amplitude, dominant_freq].
        """
        n_ep, n_ch, n_samp = epochs.shape
        out = np.zeros((n_ep, 3))
        duration_min = n_samp / sfreq / 60.0

        for i, epoch in enumerate(epochs):
            # Work on the mean across channels
            sig = epoch.mean(axis=0)
            z = (sig - sig.mean()) / (sig.std() + 1e-8)

            # Find peaks above threshold
            from scipy.signal import find_peaks
            peaks, props = find_peaks(np.abs(z), height=SPIKE_THRESHOLD_ZSCORE,
                                      distance=int(sfreq * 0.05))

            spike_rate = len(peaks) / duration_min
            if len(peaks) > 0:
                spike_amp = sig[peaks].std()
                # Dominant frequency of spike pattern via autocorrelation
                if len(peaks) >= 2:
                    ipi = np.diff(peaks) / sfreq   # inter-peak intervals in seconds
                    dom_freq = 1.0 / (np.median(ipi) + 1e-8)
                else:
                    dom_freq = 0.0
            else:
                spike_amp = 0.0
                dom_freq = 0.0

            out[i] = [spike_rate, spike_amp, dom_freq]
        return out

    def _coherence_features(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Mean coherence (global) and frontal coherence (first 2 channels).
        Returns [mean_coherence, frontal_coherence].
        """
        from scipy.signal import coherence
        n_ep, n_ch, n_samp = epochs.shape
        out = np.zeros((n_ep, 2))

        for i, epoch in enumerate(epochs):
            pairs_all = []
            pairs_frontal = []
            for a in range(n_ch):
                for b in range(a + 1, n_ch):
                    _, cxy = coherence(epoch[a], epoch[b], fs=sfreq,
                                       nperseg=min(n_samp, int(sfreq)))
                    # Delta+theta band coherence (0.5–8 Hz)
                    c_val = cxy[:int(8 * n_samp / sfreq)].mean()
                    pairs_all.append(c_val)
                    if a < 2 and b < 2:
                        pairs_frontal.append(c_val)

            out[i, 0] = np.mean(pairs_all) if pairs_all else 0.0
            out[i, 1] = np.mean(pairs_frontal) if pairs_frontal else out[i, 0]
        return out

    def _pac(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Phase-Amplitude Coupling: theta phase (4–8 Hz) → gamma amplitude (30–45 Hz).
        Uses modulation index (Tort et al.).
        """
        from scipy.signal import butter, filtfilt, hilbert
        n_ep = epochs.shape[0]
        out = np.zeros(n_ep)

        def bandpass_1d(x, lo, hi, fs):
            b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
            return filtfilt(b, a, x)

        for i, epoch in enumerate(epochs):
            sig = epoch.mean(axis=0)
            try:
                theta = bandpass_1d(sig, 4.0, 8.0, sfreq)
                gamma = bandpass_1d(sig, 30.0, 45.0, sfreq)
                phase = np.angle(hilbert(theta))
                amp   = np.abs(hilbert(gamma))

                # Modulation index: bin phase, measure amp distribution
                n_bins = 18
                bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                amp_binned = np.array([
                    amp[(phase >= bins[k]) & (phase < bins[k + 1])].mean()
                    for k in range(n_bins)
                ])
                amp_binned /= (amp_binned.sum() + 1e-8)
                # KL divergence from uniform
                mi = np.sum(amp_binned * np.log(amp_binned * n_bins + 1e-8)) / np.log(n_bins)
                out[i] = max(0.0, mi)
            except Exception:
                out[i] = 0.0
        return out

    def _extra_features(self, epochs, sfreq, band_powers):
        """
        background_slowing_index, frontal_delta_asymmetry, inter_burst_interval.
        Returns (n_ep, 3).
        """
        from scipy.signal import welch
        n_ep, n_ch, n_samp = epochs.shape
        out = np.zeros((n_ep, 3))
        nperseg = min(n_samp, int(sfreq * 2))

        for i, epoch in enumerate(epochs):
            freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
            psd_mean = psd.mean(axis=0)

            # Background slowing: fraction of power below 8 Hz
            low_mask  = freqs < 8.0
            total_pow = psd_mean.sum() + 1e-8
            out[i, 0] = psd_mean[low_mask].sum() / total_pow

            # Frontal delta asymmetry (ch0 vs ch1 if available, else 0)
            if n_ch >= 2:
                d_mask = (freqs >= 0.5) & (freqs < 4.0)
                ch0_delta = psd[0][d_mask].mean()
                ch1_delta = psd[1][d_mask].mean()
                out[i, 1] = (ch0_delta - ch1_delta) / (ch0_delta + ch1_delta + 1e-8)
            else:
                out[i, 1] = 0.0

            # Inter-burst interval: detect bursts (high-amplitude windows)
            sig = epoch.mean(axis=0)
            rms = np.sqrt(np.convolve(sig ** 2, np.ones(int(sfreq * 0.1)) / int(sfreq * 0.1), mode="same"))
            threshold = rms.mean() + rms.std()
            burst_mask = (rms > threshold).astype(int)
            transitions = np.diff(burst_mask)
            burst_starts = np.where(transitions == 1)[0]
            burst_ends   = np.where(transitions == -1)[0]
            if len(burst_starts) > 1:
                ibi = np.diff(burst_starts) / sfreq
                out[i, 2] = np.median(ibi)
            else:
                out[i, 2] = 0.0
        return out

    def _per_channel_delta(self, epochs: np.ndarray, sfreq: float) -> np.ndarray:
        """Log delta power per channel, zero-padded to MAX_CHANNELS. Returns (n_ep, MAX_CH)."""
        from scipy.signal import welch
        n_ep, n_ch, n_samp = epochs.shape
        nperseg = min(n_samp, int(sfreq * 2))
        out = np.zeros((n_ep, MAX_CHANNELS))

        for i, epoch in enumerate(epochs):
            freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
            d_mask = (freqs >= 0.5) & (freqs < 4.0)
            for c in range(min(n_ch, MAX_CHANNELS)):
                out[i, c] = np.log1p(psd[c][d_mask].mean())
        return out


# ---------------------------------------------------------------------------
# Convenience: load from EDF file without instantiating class
# ---------------------------------------------------------------------------
def features_from_edf(path: str) -> np.ndarray:
    """One-liner for use in training scripts."""
    m = EEGModality(edf_path=path)
    return m.run()