"""
"""
import os

import librosa
import librosa.feature
import numpy as np
import scipy.io
import tensorflow as tf


def inject_noise(wave, scale=1/256):
    """
    FFTNET, 2.3.3, injected noise

    we inject gaussian noise centered at 0 with a standard devaition of 1/256
    (based on 8 bit quantization).
    """
    noise = np.random.normal(loc=0.0, scale=scale, size=wave.shape)

    return wave + noise


def mu_law_encode(wave, quantization_channels=256):
    """
    FFTNET, 3.1, experimental setup,
    the waveforms are quantized to 256 categorical values based on mu-law.

    WAVENET, 2.2, softmax distribution,
    mu-law equation

    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """
    mu = float(quantization_channels - 1)

    return np.sign(wave) * np.log(1.0 + mu * np.abs(wave)) / np.log(1.0 + mu)


def mu_law_decode(wave, quantization_channels=256):
    """
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    generated_samples = np.sign(signal) * magnitude
    """
    mu = float(quantization_channels - 1)

    return np.sign(wave) * ((1.0 + mu) ** np.abs(wave) - 1.0) / mu


def quantize(wave, quantization_channels=256):
    """
    wave: wave samples between -1.0 ~ +1.0
    quantization_channels: number of categorical values
    """
    quantization_channels = float(quantization_channels - 1)

    wave = quantization_channels * 0.5 * (wave + 1.0) + 0.5

    return np.clip(wave, 0.0, quantization_channels).astype(np.int32)


def one_hot(wave, num_categories=256):
    """
    """
    results = np.zeros((wave.size, num_categories), dtype=np.float32)

    results[np.arange(wave.size), wave] = 1.0

    return results


def random_waves(source_npzs_path, samples_size):
    """
    """
    def npz_paths():
        """
        """
        names = tf.gfile.ListDirectory(source_npzs_path)
        names = [name for name in names if name.endswith('.npz')]

        while True:
            np.random.shuffle(names)

            for name in names:
                yield os.path.join(source_npzs_path, name)

    for npz_path in npz_paths():
        data = np.load(npz_path)

        wave, features = data['wave'], data['features']

        if wave.shape[0] != features.shape[0]:
            raise Exception('invalid data, miss matched data shapes')

        # NOTE: random crop
        idx = np.random.randint(wave.shape[0] - samples_size)

        features = features[idx+1:idx+samples_size+1]

        noise_wave = inject_noise(wave[idx:idx+samples_size])
        clean_wave = wave[idx+1:idx+samples_size+1]

        noise_wave = quantize(noise_wave, quantization_channels=256)
        clean_wave = quantize(clean_wave, quantization_channels=256)

        noise_wave = one_hot(noise_wave)
        clean_wave = one_hot(clean_wave)

        yield clean_wave, noise_wave, features


def wave_batches(source_npzs_path, batch_size, samples_size):
    """
    """
    waves_features = random_waves(source_npzs_path, samples_size)

    while True:
        data = [next(waves_features) for _ in range(batch_size)]

        clean_waves, noise_waves, features = zip(*data)

        clean_waves = np.stack(clean_waves, axis=0)
        noise_waves = np.stack(noise_waves, axis=0)

        features = np.stack(features, axis=0)

        yield clean_waves, noise_waves, features


def dense_features(
        wave,
        sampling_rate,
        feature_name,
        feature_size,
        fft_window_size,
        fft_hop_length):
    """
    """
    if feature_name == 'mfcc':
        features = librosa.feature.mfcc(
            y=wave,
            sr=sampling_rate,
            n_mfcc=feature_size,
            n_fft=fft_window_size,
            hop_length=fft_hop_length)
    else:
        features = librosa.feature.melspectrogram(
            y=wave,
            sr=sampling_rate,
            n_mels=feature_size,
            n_fft=fft_window_size,
            hop_length=fft_hop_length)

        features = librosa.power_to_db(features)

    features = features.T

    num_samples = (features.shape[0] - 1) * fft_hop_length

    # NOTE: FFTNET, 2.2
    #       For the ht that are not located at the window centers, we linearly
    #       interpolate their values based on the assigned ht in the last step.
    interp_features = np.zeros((num_samples, feature_size), dtype=np.float32)

    for t in range(num_samples):
        idx = t // fft_hop_length
        mod = t % fft_hop_length

        alpha = mod / fft_hop_length

        interp_features[t] = \
            (1.0 - alpha) * features[idx] + alpha * features[idx + 1]

    return interp_features


def preprocess_wave(
        source_wav_path,
        result_npz_path,
        feature_name='mfcc',
        feature_size=25,
        fft_window_size=400,
        fft_hop_length=160):
    """
    output features & mu-law results to a npz
    """
    # NOTE: read wave, assume they are 16-bits in 16,000 Hz
    # NOTE: librosa does not accept file object
    sr, wave = scipy.io.wavfile.read(source_wav_path)

    wave = wave.astype(np.float32) / 32768.0

    # NOTE: FFTNET, 2.2
    #       For the ht that are not located at the window centers, we
    #       linearly interpolate their values based on the assigned ht in
    #       the last step.
    features = dense_features(
        wave, sr, feature_name, feature_size, fft_window_size, fft_hop_length)

    # NOTE: FFTNET, 2.2
    #       For the ht corresponding to the window centers, we assign the
    #       computed MCC and F0 values (26 dimensions in total).
    #
    # NOTE: https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py
    #       melspectrogram use _spectrogram
    #
    #       https://github.com/librosa/librosa/blob/master/librosa/core/spectrum.py
    #       _spectrogram use stft with default parameters
    #       stft has a parameter center whose default is True
    #       which means:
    #       the signal `y` is padded so that frame `D[:, t]` is centered at
    #       `y[t * hop_length]`
    #       which means the returned features is aligned already.
    #
    # NOTE: skip half window size to make sure all features are build on whole
    #       size window
    features = features[fft_window_size // 2:]

    wave = wave[fft_window_size // 2:]

    # NOTE: align length
    aligned_length = min(wave.shape[0], features.shape[0])

    wave = wave[:aligned_length]

    features = features[:aligned_length]

    # NOTE: FFTNET, 3.1, experimental setup,
    #       the waveforms are quantized to 256 categorical values based on
    #       mu-law.
    wave = mu_law_encode(wave)

    # NOTE: can not use tf.gfile.GFile
    np.savez(
        result_npz_path,
        wave=wave,
        feature_name=feature_name,
        features=features)


if __name__ == '__main__':
    # NOTE: preprocess wav to mu-law encoded and features (mfcc/mel)
    # NOTE: problems on multiprocessing with numpy

    import argparse

    parser = argparse.ArgumentParser(description='preprocess wav files')

    parser.add_argument('--source_wav_dir', type=str)
    parser.add_argument('--result_npz_dir', type=str)
    parser.add_argument('--feature_name', type=str)
    parser.add_argument('--feature_bins', type=int)

    args = parser.parse_args()

    names = tf.gfile.ListDirectory(args.source_wav_dir)
    names = [name for name in names if name.endswith('.wav')]

    for idx, name in enumerate(names):
        name, ext = os.path.splitext(name)

        source_wav_path = os.path.join(args.source_wav_dir, name + ext)
        result_npz_path = os.path.join(args.result_npz_dir, name + '.npz')

        preprocess_wave(
            source_wav_path,
            result_npz_path,
            feature_name=args.feature_name,
            feature_size=args.feature_bins)

        print('done [{}]: {}/{}'.format(args.feature_name, idx, len(names)))

