import glob
from sklearn.decomposition import NMF
from denoise import *

if __name__ == '__main__':
    clean_dir = '/Users/goree/Desktop/cmu/datasets/timit/data'
    mixed_dir = '/Users/goree/Desktop/cmu/datasets/timit_snr0'
    noise_file = '/Users/goree/Desktop/cmu/datasets/noise/white_noise.wav'
    recon_dir = '/Users/goree/Desktop/cmu/recon'

    wav_mixed_files = glob.glob(os.path.join(mixed_dir, '*', '*', '*', '*.wav'))
    wav_clean_files = glob.glob(os.path.join(clean_dir, '*', '*', '*', '*.WAV.wav'))
    wav_mix_file = wav_mixed_files[0]
    wav_clean_file = wav_clean_files[0]
    
    V_mixed, phase_mixed, sr_mixed = get_magnitude_spectrum(wav_mix_file)
    V_clean, phase_clean, sr_clean = get_magnitude_spectrum(wav_clean_file)
    V_noise, phase_noise, sr_noise = get_magnitude_spectrum(noise_file)

    print(V_mixed.shape)
    print(V_clean.shape)
    print(V_noise.shape)

    model_clean = NMF(init='random', solver='mu', beta_loss='kullback-leibler', max_iter=300, alpha=0)
    W = model_clean.fit_transform(V_clean)
    H = model_clean.components_

    print("W clean shape:", W.shape)
    print("H clean shape:", H.shape)

    # test if learned bases and weights are good
    clean_rec = librosa.istft(W @ H * phase_clean, hop_length=256, center=False, win_length=2048)
    sf.write(os.path.join(recon_dir, 'clean_recon_test.wav'), clean_rec, samplerate=sr_clean)
