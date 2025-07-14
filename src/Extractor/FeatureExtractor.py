import logging
import librosa as lr
import numpy as np

from src.Utils.Logger import get_logger

class FeatureExtractor:
    """
    Class for extracting voice-related features from audio files,
    specifically PPQ55 jitter based on estimated F0 (fundamental frequency) values.
    """

    def __init__(self):
        self.Logger: logging = get_logger(__name__)  # Initialize logger for this class/module
        self.universal_fmin = 65  # Minimum frequency (Hz) for pitch estimation (e.g., ~C2)
        self.universal_fmax = 2093  # Maximum frequency (Hz) for pitch estimation (e.g., ~C7)

    def get_f0_lens(self, samples: np.ndarray) -> np.ndarray:
        """
        Estimate the fundamental frequency (f0) using librosa's YIN algorithm,
        then return its period (i.e., 1/f0) to use for jitter calculation.

        Args:
            samples (np.ndarray): Audio signal samples.

        Returns:
            np.ndarray: Array of F0 periods (1/f0).
        """
        f0 = lr.yin(samples, fmin=self.universal_fmin, fmax=self.universal_fmax)
        return 1 / f0  # Convert frequency to period

    def get_ppq55_jitter(self, filepath: str) -> float:
        """
        Calculate the PPQ55 jitter for a given audio file.
        PPQ55 is a measure of cycle-to-cycle variation in pitch period over a 55-point window.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            float: Computed PPQ55 jitter percentage.
        """

        samples = self.load_samples(filepath)
        f0_lens: np.ndarray = self.get_f0_lens(samples)  # Get pitch periods
        N: int = len(f0_lens)
        difsum: float = 0

        # Sliding window approach: calculate absolute difference from local average over 55 frames
        for i in range(27, N - 27):
            difsum += np.abs(f0_lens[i] - np.average(f0_lens[i - 27:i + 28]))

        # Normalize and convert to percentage
        ppq55_jitter: float = ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

        return ppq55_jitter

    @staticmethod
    def load_samples(filepath: str) -> np.ndarray:
        """
        Load audio samples from a .wav file using librosa.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            np.ndarray: Array of audio samples.
        """
        sr: float = lr.get_samplerate(filepath)  # Get native sample rate
        samples, _ = lr.load(filepath, sr=sr)  # Load audio with original sample rate
        return samples
