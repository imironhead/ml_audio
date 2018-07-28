"""
"""
import numpy as np
import os
import scipy.io.wavfile
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
    zeros = np.zeros((size_zeros,), dtype=wave.dtype)

    return np.concatenate([zeros, wave])


def inject_noise(wave):
    """
    FFTNET, 2.3.3, injected noise

    we inject gaussian noise centered at 0 with a standard devaition of 1/256
    (based on 8 bit quantization).
    """
    noise = np.random.normal(loc=0.0, scale=1/512, size=wave.shape)

    return wave + noise


def mu_law(wave):
    """
    FFTNET, 3.1, experimental setup,
    the waveforms are quantized to 256 categorical values based on mu-law.

    WAVENET, 2.2, softmax distribution,
    mu-law equation
    """
    wave = np.sign(wave) * np.log(1.0 + 255.0 * np.abs(wave)) / np.log(256.0)

    wave = 255.0 * 0.5 * (wave + 1.0) + 0.5

    return np.clip(wave, 0.0, 255.0).astype(np.int32)


def random_waves(dir_path, samples_size, zeros_size):
    """
    """
    names = tf.gfile.ListDirectory(dir_path)
    names = [name for name in names if name.endswith('.wav')]

    while True:
        np.random.shuffle(names)

        for name in names:
            # NOTE: read wave, assume they are 16-bits in 16,000 Hz
            wave_path = os.path.join(dir_path, name)

            with tf.gfile.GFile(wave_path, 'rb') as wave_file:
                _, wave = scipy.io.wavfile.read(wave_file)

            # NOTE: sanity check, reject short wave data
            if wave.size < samples_size:
                continue

            # NOTE: FFTNET, 3.1
            #       - in each training step, a minibatch of 5x5000 sample
            #         sequences are fed to the network.
            #       - in each minibatch, all sequences come from different
            #         utterances.
            wave = random_crop(wave, samples_size)

            wave = wave / 32768.0

            # NOTE: FFTNET, 2.3.1
            wave = pad_zero(wave, zeros_size)

            # NOTE: FFTNET, 2.3.3
            wave = inject_noise(wave)

            # NOTE: FFTNET, 3.1
            wave = mu_law(wave)

            yield wave


def wave_batches(dir_path, batch_size, samples_size, zeros_size):
    """
    """
    waves = random_waves(dir_path, samples_size, zeros_size)

    while True:
        yield [next(waves).reshape(1, -1, 1) for _ in range(batch_size)]

