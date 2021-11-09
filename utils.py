import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os


def get_magnitude_spectrum(path, sr=None):
    audio, sample_rate = librosa.load(path, sr=sr)
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048)
    M = abs(spectrogram)
    phase = spectrogram / (M + 2.2204e-16)

    return M, phase, sample_rate


def calculate_kldiv(A, B):
    divergence = np.sum(np.multiply(A, np.log10(np.divide(A, B))) - A + B)
    return divergence


def get_file_name(file_path):
    basename = os.path.basename(file_path)
    filename = basename.split('.')[0]
    return filename + ".wav"
