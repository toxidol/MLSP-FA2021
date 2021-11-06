import glob
from denoise import *

if __name__ == '__main__':
    mixed_dir = '/Users/goree/Desktop/cmu/datasets/timit_snr0'

    wav_files = glob.glob(os.path.join(mixed_dir, '*', '*', '*', '*.wav'))
    wav_file = wav_files[0]
    print(wav_file)
