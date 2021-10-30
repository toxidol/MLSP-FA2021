'''
Add noise to audio files in TIMIT.
'''

import os
import glob
import random
from tqdm import tqdm
from create_mixed_audio_file_with_soundfile import mix_clean_with_noise


if __name__ == '__main__':
    snr = 0     # in dB
    timit_root = '/data-ssd/Datasets/timit/data'
    output_root = f'/data-ssd/Datasets/timit_snr{snr}'
    noise_files = [
        '/data-ssd/Datasets/noise/white_noise.wav'
    ]

    wav_files = glob.glob(os.path.join(timit_root, '*', '*', '*', '*.WAV.wav'))

    for cur_wav in tqdm(wav_files):
        cur_dir = '/'.join([output_root] + os.path.dirname(cur_wav).split('/')[-3:])
        cur_id = os.path.basename(cur_wav)[:-8]
        os.makedirs(cur_dir, exist_ok=True)
        mix_clean_with_noise(
            cur_wav,
            random.choice(noise_files),
            snr,
            os.path.join(cur_dir, f"{cur_id}.wav")
        )
        os.system(
            f'cp {os.path.join(os.path.dirname(cur_wav), f"{cur_id}.TXT")} {os.path.join(cur_dir, f"{cur_id}.txt")}'
        )
