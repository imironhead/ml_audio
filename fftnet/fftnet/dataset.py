"""
"""
import os

import librosa
import librosa.feature
import numpy as np
import scipy.io
import tensorflow as tf


def random_crop(wave, size):
    """
    FFTNET, 3.1
    in each training step, a minibatch of 5x5000 sample sequences are fed to
    the network, ...

    random crop wave to size
    """
    idx = np.random.randint(wave.shape[0] - size + 1)

    return wave[idx:idx+size]


def pad_zero(wave, size_zeros):
    """
    FFTNET, 2.3.1, zero padding,
    we recommend using training sequences of length between 2N and 3N so that
    significant number (33% - 50%) of training samples are partial sequences.

    FFTNET, 3.1, experimental setup,
    the FFTNet implementation contains 11 FFT-layers with 256 channels, ...

    FFTNET, 3.1, experimental setup,
    in each training step, a minibatch of 5x5000 sample sequences are fed to
    the network.

    => assume N is 2048, 5000 samples are 2.44 N.
    """
    if size_zeros > 0:
        zeros = np.zeros((size_zeros,), dtype=wave.dtype)

        return np.concatenate([zeros, wave])
    else:
        return wave


def inject_noise(wave):
    """
    FFTNET, 2.3.3, injected noise

    we inject gaussian noise centered at 0 with a standard devaition of 1/256
    (based on 8 bit quantization).
    """
    noise = np.random.normal(loc=0.0, scale=1/256, size=wave.shape)

    return wave + noise


def dense_mfccs(wave, sampling_rate, num_mfcc, window_size, hop_length):
    """
    """
    mfccs = librosa.feature.mfcc(
        y=wave,
        sr=sampling_rate,
        n_mfcc=num_mfcc,
        n_fft=window_size,
        hop_length=hop_length)

#   mfccs = librosa.feature.melspectrogram(
#       y=wave,
#       sr=sampling_rate,
#       n_mels=num_mfcc,
#       n_fft=window_size,
#       hop_length=hop_length)

#   mfccs = librosa.power_to_db(mfccs)

    mfccs = mfccs.T

    num_samples = (mfccs.shape[0] - 1) * hop_length

    # NOTE: FFTNET, 2.2
    #       For the ht that are not located at the window centers, we
    #       linearly interpolate their values based on the assigned ht in
    #       the last step.
    interp_mfccs = np.zeros((num_samples, num_mfcc), dtype=np.float32)

    for t in range(num_samples):
        idx = t // hop_length
        mod = t % hop_length

        alpha = mod / hop_length

        interp_mfccs[t] = (1.0 - alpha) * mfccs[idx] + alpha * mfccs[idx + 1]

    # TODO: how to normalize it?

    return interp_mfccs / 800.0


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


def _random_waves(
        dir_path,
        samples_size,
        zeros_size,
        num_mfccs=20,
        window_size=400,
        hop_length=160):
    """
    """
    def wave_paths():
        """
        """
        names = tf.gfile.ListDirectory(dir_path)
        names = [name for name in names if name.endswith('.wav')]

        while True:
            np.random.shuffle(names)

            for name in names:
                yield os.path.join(dir_path, name)

    # NOTE: extend samples_size as temp_samples_size to make sure the mfcc are
    #       well interpolated
    # NOTE: +2 for interpolation of the latest segment
    temp_samples_size = (samples_size + zeros_size) // hop_length + 2
    temp_samples_size = temp_samples_size * hop_length + window_size
    temp_samples_size = temp_samples_size - zeros_size

    output_size = samples_size + zeros_size

    for wave_path in wave_paths():
        # NOTE: read wave, assume they are 16-bits in 16,000 Hz
        # NOTE: librosa does not accept file object
        with tf.gfile.GFile(wave_path, 'rb') as wave_file:
            sr, wave = scipy.io.wavfile.read(wave_file)

        # NOTE: sanity check, reject short wave data
        if wave.size < temp_samples_size:
            continue

        # NOTE: FFTNET, 3.1
        #       - in each training step, a minibatch of 5x5000 sample
        #         sequences are fed to the network.
        #       - in each minibatch, all sequences come from different
        #         utterances.
        wave = random_crop(wave, temp_samples_size)

        # NOTE: FFTNET, 2.3.1
        wave = pad_zero(wave, zeros_size)

        wave = wave.astype(np.float32) / 32768.0

        # NOTE: FFTNET, 2.2
        #       For the ht that are not located at the window centers, we
        #       linearly interpolate their values based on the assigned ht in
        #       the last step.
        mfccs = dense_mfccs(wave, sr, num_mfccs, window_size, hop_length)

        # NOTE: FFTNET, 2.2
        #       For the ht corresponding to the window centers, we assign the
        #       computed MCC and F0 values (26 dimensions in total).
        wave = wave[window_size // 2:]

        # NOTE: FFTNET, 2.3.3
        noise_wave = np.copy(wave)
        clean_wave = wave

        # NOTE: FFTNET, 3.1
        noise_wave = mu_law_encode(noise_wave)
        clean_wave = mu_law_encode(clean_wave)

        noise_wave = inject_noise(noise_wave)

        noise_wave = quantize(noise_wave, quantization_channels=256)
        clean_wave = quantize(clean_wave, quantization_channels=256)

        noise_wave = one_hot(noise_wave)
        clean_wave = one_hot(clean_wave)

        noise_wave = noise_wave[:output_size]
        clean_wave = clean_wave[:output_size]

        mfccs = mfccs[:output_size]

        yield clean_wave, noise_wave, mfccs


def _wave_batches(
        dir_path,
        batch_size,
        samples_size,
        zeros_size,
        num_mfccs=20,
        window_size=400,
        hop_length=160):
    """
    """
    waves_mfccs = random_waves(
        dir_path, samples_size, zeros_size, num_mfccs, window_size, hop_length)

    while True:
        data = [next(waves_mfccs) for _ in range(batch_size)]

        clean_waves, noise_waves, mfccs = zip(*data)

        clean_waves = np.stack(clean_waves, axis=0)
        noise_waves = np.stack(noise_waves, axis=0)

        mfccs = np.stack(mfccs, axis=0)

        yield clean_waves, noise_waves, mfccs


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

    args = parser.parse_args()

    names = tf.gfile.ListDirectory(args.source_wav_dir)
    names = [name for name in names if name.endswith('.wav')]

    for idx, name in enumerate(names):
        name, ext = os.path.splitext(name)

        source_wav_path = os.path.join(args.source_wav_dir, name + ext)
        result_npz_path = os.path.join(args.result_npz_dir, name + '.npz')

        preprocess_wave(
            source_wav_path, result_npz_path, feature_name=args.feature_name)

        print('done [{}]: {}/{}'.format(args.feature_name, idx, len(names)))

