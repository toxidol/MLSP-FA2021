import glob
from sklearn.decomposition import NMF, non_negative_factorization
from denoise import *

if __name__ == '__main__':
    clean_dir = '/Users/goree/Desktop/cmu/datasets/timit/data'
    mixed_dir = '/Users/goree/Desktop/cmu/datasets/timit_snr0'
    noise_file = '/Users/goree/Desktop/cmu/datasets/noise/white_noise.wav'
    recon_dir = '/Users/goree/Desktop/cmu/recon_reg'

    wav_mixed_files = glob.glob(os.path.join(mixed_dir, '*', '*', '*', '*.wav'))
    wav_clean_files = glob.glob(os.path.join(clean_dir, '*', '*', '*', '*.WAV.wav'))

    recon(wav_mixed_files, wav_clean_files, noise_file, recon_dir)
