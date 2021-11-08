"""
Install pysepm as follows:
    pip3 install https://github.com/schmiph2/pysepm/archive/master.zip
"""

import soundfile as sf
import pysepm
import numpy as np


if __name__ == '__main__':
    clean_speech, fs = sf.read('eval/SA1_clean.wav')
    noisy_speech, fs = sf.read('eval/SA1_noisy.wav')
    enhanced_speech, fs = sf.read('eval/SA1_enh.wav')

    # ensure all signals have the same length
    N = min([len(enhanced_speech), len(clean_speech), len(noisy_speech)])
    clean_speech = clean_speech[:N]
    noisy_speech = noisy_speech[:N]
    enhanced_speech = enhanced_speech[:N]

    # normalize all signals
    clean_speech = clean_speech / np.abs(clean_speech).max()
    noisy_speech = noisy_speech / np.abs(noisy_speech).max()
    enhanced_speech = enhanced_speech / np.abs(enhanced_speech).max()

    snr_before_enh = pysepm.SNRseg(clean_speech, noisy_speech, fs)
    snr_after_enh = pysepm.SNRseg(clean_speech, enhanced_speech, fs)

    print(f'Segmental SNR before enhancement: {snr_before_enh:.4f}')
    print(f'Segmental SNR after enhancement: {snr_after_enh:.4f}')
