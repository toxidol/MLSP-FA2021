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
    timit_root = '/Users/goree/Desktop/cmu/datasets/timit/data'  # change for user path structure
    output_root = f'/Users/goree/Desktop/cmu/datasets/timit_snr{snr}'  # change for user path structure

    # change for user path structure
    noise_files = [
        '/Users/goree/Desktop/cmu/datasets/noise/white_noise.wav'
    ]

    wav_files = glob.glob(os.path.join(timit_root, '*', '*', '*', '*.WAV.wav'))

    for cur_wav in tqdm(wav_files):
        # cur_dir = '/'.join([output_root] + os.path.dirname(cur_wav).split('/')[-3:])
        cur_dir = os.path.join(output_root, *os.path.dirname(cur_wav).split('\\')[-3:])  # windows usage
        cur_id = os.path.basename(cur_wav)[:-8]
        os.makedirs(cur_dir, exist_ok=True)
        mix_clean_with_noise(
            cur_wav,
            random.choice(noise_files),
            snr,
            os.path.join(cur_dir, f"{cur_id}.wav")
        )

        # this cp_cmd is for windows usage. Change 'copy' to 'cp' for unix systems
        # also remove string quotes around source dest files in unix
        cp_cmd = f'copy "{os.path.join(os.path.dirname(cur_wav), f"{cur_id}.TXT")}" "{os.path.join(cur_dir, f"{cur_id}.txt")}" >NUL'
        os.system(cp_cmd)
