import glob
from sklearn.decomposition import NMF, non_negative_factorization
from denoise import *

if __name__ == '__main__':
    clean_dir = '/Users/goree/Desktop/cmu/datasets/timit/data'
    mixed_dir = '/Users/goree/Desktop/cmu/datasets/timit_snr0'
    noise_file = '/Users/goree/Desktop/cmu/datasets/noise/white_noise.wav'
    recon_dir = '/Users/goree/Desktop/cmu/recon'

    wav_mixed_files = glob.glob(os.path.join(mixed_dir, '*', '*', '*', '*.wav'))
    wav_clean_files = glob.glob(os.path.join(clean_dir, '*', '*', '*', '*.WAV.wav'))
    # wav_mix_file = wav_mixed_files[0]
    wav_clean_file = wav_clean_files[0]
    
    # V_mixed, phase_mixed, sr_mixed = get_magnitude_spectrum(wav_mix_file)
    V_clean, phase_clean, sr_clean = get_magnitude_spectrum(wav_clean_file)
    V_noise, phase_noise, sr_noise = get_magnitude_spectrum(noise_file)

    # print(V_mixed.shape)
    print("shape of clean signal:", V_clean.shape)
    print("shape of noisy signal:", V_noise.shape)

    model_clean = NMF(init='random', solver='mu', beta_loss='kullback-leibler', max_iter=500, alpha=0)
    W_clean = model_clean.fit_transform(V_clean)  # bases
    H_clean = model_clean.components_

    print("W clean shape:", W_clean.shape)
    print("H clean shape:", H_clean.shape)

    model_noise = NMF(init='random', solver='mu', beta_loss='kullback-leibler', max_iter=500, alpha=0)
    W_noise = model_noise.fit_transform(V_noise)  # bases
    H_noise = model_noise.components_

    print("W noise shape:", W_noise.shape)
    print("H noise shape:", H_noise.shape)

    # test if learned bases and weights are good
    # clean_rec = librosa.istft(W @ H * phase_clean, hop_length=256, center=False, win_length=2048)
    # sf.write(os.path.join(recon_dir, 'clean_recon_test.wav'), clean_rec, samplerate=sr_clean)

    test_mixed_file = wav_mixed_files[0]
    print("mixed file:", test_mixed_file)
    V_mixed, phase_mixed, sr_mixed = get_magnitude_spectrum(test_mixed_file)
    print("shape of mixed signal:", V_mixed.shape)

    W = np.concatenate([W_clean, W_noise], axis=1)
    # model_mixed = NMF(n_components=W.shape[1], init='random', solver='mu', beta_loss='kullback-leibler', max_iter=500, alpha=0)
    H_mixed, W_mixed, n_iter = non_negative_factorization(V_mixed.T, H=W.T, n_components=W.shape[1], init='random', update_H=False, solver='mu', beta_loss='kullback-leibler', max_iter=400)
    p, k = W_clean.shape
    
    # W_mixed = model_mixed.fit_transform(V_mixed)  # don't use W_mixed
    # H_mixed = model_mixed.components_

    print("got weights for mixed signal")

    assert(W_mixed.T.shape == W.shape)
    H_clean_recon = H_mixed.T[:k, :]

    mixed_rec = librosa.istft(W_clean @ H_clean_recon * phase_mixed, hop_length=256, center=False, win_length=2048)
    sf.write(os.path.join(recon_dir, 'clean_recon_test.wav'), mixed_rec, samplerate=sr_mixed)
