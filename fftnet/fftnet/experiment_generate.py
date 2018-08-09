"""
"""
import os

import numpy as np
import scipy.io.wavfile
import tensorflow as tf

import fftnet.dataset as dataset
import fftnet.model_fftnet as model_fftnet


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = model_fftnet.build_model(
        False,
        2048,
        2048,
        sample_size=FLAGS.wave_quantization_size,
        condition_size=FLAGS.num_mfccs,
        hidden_size=FLAGS.hidden_size,
        num_layers=FLAGS.num_layers)

    return model


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_model()

#   source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)

    num_source_samples = 2 ** FLAGS.num_layers

    ###
    with tf.gfile.GFile(FLAGS.source_wave_path, 'rb') as wave_file:
        sr, wave = scipy.io.wavfile.read(wave_file)

    wave = wave.astype(np.float32) / 32768.0

    wave = np.pad(wave, [2248, 0], 'constant', constant_values=[0.0, 0.0])

    mfccs = dataset.dense_mfccs(
        wave, sr, FLAGS.num_mfccs, FLAGS.fft_window_size, FLAGS.fft_hop_length)

    # NOTE: FFTNET, 2.2
    #       For the ht corresponding to the window centers, we assign the
    #       computed MCC and F0 values (26 dimensions in total).
    wave = wave[FLAGS.fft_window_size // 2:]

    # NOTE: FFTNET, 3.1
    wave = dataset.mu_law(wave, False)

    wave = dataset.one_hot(wave)

    final_size = min(wave.shape[0], mfccs.shape[0])

    wave = wave[:final_size, :].reshape(1, final_size, FLAGS.wave_quantization_size)

    mfccs = mfccs[:final_size, :].reshape(1, final_size, FLAGS.num_mfccs)

    wave[:, 2048:, :] = 0.0

    ###
    generated_samples = []

    with tf.Session() as session:
        saver = tf.train.Saver()

        tf.train.Saver().restore(session, FLAGS.ckpt_path)

        for i in range(FLAGS.len_seconds * 16000):
            if i % 16000 == 0:
                print('[{}]'.format(i))

            feeds = {
                model['source_waves']: wave[:, i:i+2048, :],
                model['condition_tensors']: mfccs[:, i+1:i+2049, :],
            }

            result_waves = session.run(model['result_waves'], feed_dict=feeds)

#           if i % 100 == 0:
#               print('{} - {}'.format(np.argmax(result_waves), np.max(result_waves)))

            m = np.random.choice(np.arange(256), p=result_waves[0, -1, :])

#           m = np.argmax(result_waves[0, -1, :])

#           print(m - np.argmax(wave[0, i+2048]))

            wave[0, i+2048, m] = 1.0

            generated_samples.append(m)

    generated_samples = np.array(generated_samples, dtype=np.float32)

    qc = 256
    mu = qc - 1
    wav = generated_samples
    # Map values back to [-1, 1].
    casted = wav.astype(np.float32)
    signal = 2 * (casted / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    generated_samples = np.sign(signal) * magnitude


#   generated_samples = generated_samples / 128.0 - 1.0
    generated_samples = generated_samples * 32767.0

    generated_samples = np.clip(generated_samples, -32767.0, 32767.0).astype(np.int16)

    scipy.io.wavfile.write(FLAGS.result_wave_path, 16000, generated_samples)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('source_wave_path', None, '')
    tf.app.flags.DEFINE_string('result_wave_path', None, '')

    tf.app.flags.DEFINE_integer('len_seconds', 2, '')

    # NOTE: FFTNET, 3.1, experimental setup,
    #       the waveforms are quantized to 256 categorical values based on
    #       mu-law.
    tf.app.flags.DEFINE_integer('wave_quantization_size', 256, '')

    # NOTE: FFTNET, 2.2
    #       Then we extract the MCC and F0 features for each overlapping
    #       window. For the ht corresponding to the window centers, we assign
    #       the computed MCC and F0 values (26 dimensions in total).
    # NOTE: I do not know how to get F0, use mfcc instead.
    tf.app.flags.DEFINE_integer('num_mfccs', 20, '')

    # NOTE: FFTNET, 2.2
    #       In our experiments, the auxiliary condition is obtained as follows:
    #       first we take an analysis window of size 400 every 160 samples.
    tf.app.flags.DEFINE_integer('fft_window_size', 400, '')
    tf.app.flags.DEFINE_integer('fft_hop_length', 160, '')

    # NOTE: FFTNET, 3.1
    #       The FFTNet implementation contains 11 FFT-layers with 256 channels,
    #       which also has a receptive field of 2048.
    tf.app.flags.DEFINE_integer('num_layers', 11, '')
    tf.app.flags.DEFINE_integer('hidden_size', 256, '')

    tf.app.run()

