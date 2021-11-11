import sys
import os
p = os.path.abspath('..')
sys.path.append(p)

from utils import *
from sklearn.decomposition import NMF, non_negative_factorization
from tqdm import tqdm
import glob


def NMF_train(M, B_init, W_init, n_iter):
    B = B_init
    W = W_init

    p, q = M.shape

    for _ in range(n_iter):
        B_numerator = np.divide(M, B @ W) @ W.T
        B_denominator = np.ones((p, q)) @ W.T
        B_div = np.divide(B_numerator, B_denominator)
        B = np.multiply(B, B_div)

        W_numerator = B.T @ np.divide(M, B @ W)
        W_denominator = B.T @ np.ones((p, q))
        W_div = np.divide(W_numerator, W_denominator)
        W = np.multiply(W, W_div)

    assert(B.shape == B_init.shape)
    assert(W.shape == W_init.shape)

    return B, W


def get_speech_signal(V_mixed, B_speech, B_noise, n_iter):
    # first learn W
    assert(B_speech.shape == B_noise.shape)
    p, k = B_speech.shape

    B = np.concatenate([B_speech, B_noise], axis=1)
    assert(B.shape == (p, 2 * k))

    p, q = V_mixed.shape
    W = np.random.rand(2 * k, q)

    for _ in range(n_iter):
        W_numerator = B.T @ np.divide(V_mixed, B @ W)
        W_denominator = B.T @ np.ones((p, q))
        W_div = np.divide(W_numerator, W_denominator)
        W = np.multiply(W, W_div)

    assert(W.shape == (2 * k, q))

    # reconstruction with learned H
    W_speech = W[:k, :]

    V_speech_rec = B_speech @ W_speech

    return V_speech_rec


def recon(mixed_files, clean_files, noise_file, recon_dir):
    assert(len(mixed_files) == len(clean_files))
    V_noise, phase_noise, sr_noise = get_magnitude_spectrum(noise_file)

    # learn bases for noisy signal
    print("Learning bases for noisy signal")
    model_noise = NMF(init='random', solver='mu', beta_loss='kullback-leibler', max_iter=500, alpha=0)
    W_noise = model_noise.fit_transform(V_noise)  # bases
    H_noise = model_noise.components_
    print(f"noise recon error {model_noise.reconstruction_err_}")

    total_recon_error = 0.0

    for clean_file, mixed_file in zip(clean_files, mixed_files):
        clean_filename = get_file_name(clean_file)
        print(f"RECONSTRUCTION FOR FILE {clean_filename} IN PROGRESS")
        V_clean, phase_clean, sr_clean = get_magnitude_spectrum(clean_file)

        # learn bases for clean audio signal
        model_clean = NMF(init='random', solver='mu', beta_loss='kullback-leibler', max_iter=500, alpha=0)
        W_clean = model_clean.fit_transform(V_clean)  # bases
        H_clean = model_clean.components_
        print(f"clean recon error {model_clean.reconstruction_err_}")
        total_recon_error += model_clean.reconstruction_err_

        # MIXED
        V_mixed, phase_mixed, sr_mixed = get_magnitude_spectrum(mixed_file)

        W = np.concatenate([W_clean, W_noise], axis=1)
        H_mixed, W_mixed, n_iter = non_negative_factorization(V_mixed.T, H=W.T, n_components=W.shape[1], init='custom', update_H=False, solver='mu', beta_loss='kullback-leibler', max_iter=1000)
        p, k = W_clean.shape

        assert(W_mixed.T.shape == W.shape)
        H_clean_recon = H_mixed.T[:k, :]

        mixed_rec = librosa.istft(W_clean @ H_clean_recon * phase_mixed, hop_length=256, center=False, win_length=2048)
        
        path = os.path.normpath(clean_file)
        path_dir = os.path.join(*path.split(os.sep)[-4:-1])
        write_dir = os.path.join(recon_dir, path_dir)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        write_path = os.path.join(write_dir, clean_filename)
        
        sf.write(os.path.join(write_path), mixed_rec, samplerate=sr_mixed)
        print()

    print(f"average clean reconstruction error {total_recon_error / len(clean_files)}")
