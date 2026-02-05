"""
These parameters are industry-standard:

sr=16000 → ASR norm

n_fft=400 → 25 ms window

hop_length=160 → 10 ms stride

n_mels=80 → Whisper / Conformer style"""


import librosa
import numpy as np


def extract_log_mel(
    audio_path: str,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80
):
    """
    Extract Log-Mel Spectrogram from an audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    sr : int
        Target sampling rate (default: 16 kHz)
    n_fft : int
        FFT window size (25 ms @ 16 kHz)
    hop_length : int
        Frame hop length (10 ms @ 16 kHz)
    n_mels : int
        Number of Mel filter banks

    Returns
    -------
    log_mel : np.ndarray
        Log-Mel spectrogram of shape (n_mels, time_frames)
    """

    # Load audio (mono, resampled)
    signal, sr = librosa.load(audio_path, sr=sr)

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    # Log compression
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel


def pad_log_mel(log_mel: np.ndarray, max_frames: int):
    """
    Pad or truncate a Log-Mel spectrogram to a fixed length.

    Parameters
    ----------
    log_mel : np.ndarray
        Log-Mel spectrogram (n_mels, time_frames)
    max_frames : int
        Target number of frames

    Returns
    -------
    padded : np.ndarray
        Padded Log-Mel spectrogram (n_mels, max_frames)
    mask : np.ndarray
        Binary mask indicating valid frames (1 = real, 0 = padded)
    """

    n_mels, frames = log_mel.shape

    padded = np.zeros((n_mels, max_frames), dtype=np.float32)
    mask = np.zeros(max_frames, dtype=np.float32)

    length = min(frames, max_frames)
    padded[:, :length] = log_mel[:, :length]
    mask[:length] = 1.0

    return padded, mask
