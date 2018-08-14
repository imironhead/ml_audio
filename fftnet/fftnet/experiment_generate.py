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

    model = model_fftnet.build_generating_model(
        num_quantization_levels=FLAGS.num_quantization_levels,
        condition_size=FLAGS.condition_size,
        num_layers=FLAGS.num_layers)

    return model


def prepare_conditions():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    with tf.gfile.GFile(FLAGS.source_wave_path, 'rb') as wave_file:
        sr, wave = scipy.io.wavfile.read(wave_file)

    wave = wave.astype(np.float32) / 32768.0

    conditions = dataset.dense_features(
        wave,
        sr,
        FLAGS.feature_name,
        FLAGS.condition_size,
        FLAGS.fft_window_size,
        FLAGS.fft_hop_length)

    conditions = np.pad(
        conditions,
        [[2 ** FLAGS.num_layers, 0], [0, 0]],
        'constant',
        constant_values=[[0.0, 0.0], [0.0, 0.0]])

    conditions = conditions / FLAGS.condition_scale

    return conditions


def prepare_source_queues():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    queues = []

    for i in range(FLAGS.num_layers, -1, -1):
        queue = np.zeros((2 ** i, FLAGS.num_quantization_levels), dtype=np.float32)

        queue[:, FLAGS.num_quantization_levels // 2] = 1.0

        queues.append(queue)

    return queues


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    conditions = prepare_conditions()

    source_queues = prepare_source_queues()

    model = build_model()

    l_source_tensors = np.zeros(
        (FLAGS.num_layers + 1, FLAGS.num_quantization_levels), dtype=np.float32)

    l_condition_tensors = np.zeros(
        (FLAGS.num_layers + 1, FLAGS.condition_size), dtype=np.float32)

    # NOTE: 0 ~ 255
    generated_samples = [128]

    with tf.Session() as session:
        ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)

        tf.train.Saver().restore(session, ckpt_path)

        # NOTE: skip padded zeros
        t = 2 ** FLAGS.num_layers

        while t < len(conditions):
            if len(generated_samples) % 100 == 0:
                print('generating samples[{}]'.format(len(generated_samples)))

            # NOTE: previous generated sample
            current_source_tensor = np.zeros(
                (1, FLAGS.num_quantization_levels), dtype=np.float32)

            current_source_tensor[0, generated_samples[-1]] = 1.0

            # NOTE: next condition for generating next sample
            current_condition_tensor = conditions[t:t+1]

            # NOTE: build source & condition tensors
            for i in range(FLAGS.num_layers, -1, -1):
                index = FLAGS.num_layers - i

                l_source_tensors[index] = source_queues[index][0]
                l_condition_tensors[index] = conditions[t - 2 ** i]

            # NOTE: advance on timeline
            t = t + 1

            feeds = {
                model['current_source_tensor']: current_source_tensor,
                model['current_condition_tensor']: current_condition_tensor,
                model['l_source_tensors']: l_source_tensors,
                model['l_condition_tensors']: l_condition_tensors,
            }

            fetch = {
                'result_tensor': model['result_tensor'],
                'r_source_tensors': model['r_source_tensors'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            # NOTE: sample the generated sample
            m = np.random.choice(
                np.arange(FLAGS.num_quantization_levels),
                p=fetched['result_tensor'][0])

            generated_samples.append(m)

            # NOTE: push newly generated intermediate data into queues
            for i in range(FLAGS.num_layers, -1, -1):
                index = FLAGS.num_layers - i

                source_queues[index][:-1] = source_queues[index][1:]
                source_queues[index][-1] = fetched['r_source_tensors'][index]

    # NOTE: decode generated samples
    wave = np.array(generated_samples, dtype=np.float32)

    wave = (2.0 / (FLAGS.num_quantization_levels - 1)) * wave - 1.0

    wave = dataset.mu_law_decode(wave, FLAGS.num_quantization_levels)

    wave = np.clip(wave * 32767.0, -32767.0, 32767.0).astype(np.int16)

    scipy.io.wavfile.write(FLAGS.result_wave_path, 16000, wave)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('source_wave_path', None, '')
    tf.app.flags.DEFINE_string('result_wave_path', None, '')
    tf.app.flags.DEFINE_string('feature_name', None, '')

    # NOTE: FFTNET, 3.1, experimental setup,
    #       the waveforms are quantized to 256 categorical values based on
    #       mu-law.
    tf.app.flags.DEFINE_integer('num_quantization_levels', 256, '')

    # NOTE: FFTNET, 2.2
    #       Then we extract the MCC and F0 features for each overlapping
    #       window. For the ht corresponding to the window centers, we assign
    #       the computed MCC and F0 values (26 dimensions in total).
    # NOTE: I do not know how to get F0, use mfcc instead.
    tf.app.flags.DEFINE_integer('condition_size', 20, '')

    tf.app.flags.DEFINE_float('condition_scale', 1000.0, '')

    # NOTE: FFTNET, 2.2
    #       In our experiments, the auxiliary condition is obtained as follows:
    #       first we take an analysis window of size 400 every 160 samples.
    tf.app.flags.DEFINE_integer('fft_window_size', 400, '')
    tf.app.flags.DEFINE_integer('fft_hop_length', 160, '')

    # NOTE: FFTNET, 3.1
    #       The FFTNet implementation contains 11 FFT-layers with 256 channels,
    #       which also has a receptive field of 2048.
    tf.app.flags.DEFINE_integer('num_layers', 11, '')

    tf.app.run()

