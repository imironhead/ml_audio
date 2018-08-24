"""
"""
import os
import time

import numpy as np
import scipy.io.wavfile
import tensorflow as tf

import fftnet.dataset as dataset
import fftnet.model_fftnet as model_fftnet


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = model_fftnet.build_decoding_model(
        num_quantization_levels=FLAGS.num_quantization_levels,
        condition_size=FLAGS.condition_size,
        num_layers=FLAGS.num_layers,
        logits_scaling_factor=FLAGS.logits_scaling_factor)

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


def numpy_fftnet(
        variables, source_tensor, source_queues, conditions, t):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    for i in range(FLAGS.num_layers, -1, -1):
        index = FLAGS.num_layers - i

        offset_source = t % source_queues[index].shape[0]

        offset_condition = t - 2 ** i

        result_tensor = \
            variables['fft_{}/dense_l_samples/bias:0'.format(i)].copy()

        result_tensor += np.matmul(
            source_queues[index][offset_source:offset_source+1],
            variables['fft_{}/dense_l_samples/kernel:0'.format(i)])

        result_tensor += np.matmul(
            source_tensor,
            variables['fft_{}/dense_r_samples/kernel:0'.format(i)])

        result_tensor += np.matmul(
            conditions[offset_condition:offset_condition+1],
            variables['fft_{}/dense_l_conditions/kernel:0'.format(i)])

        result_tensor += np.matmul(
            conditions[t:t+1],
            variables['fft_{}/dense_r_conditions/kernel:0'.format(i)])

        result_tensor = np.maximum(result_tensor, 0.0, result_tensor)

        result_tensor = np.matmul(
            result_tensor,
            variables['fft_{}/dense_merge/kernel:0'.format(i)])

        result_tensor += variables['fft_{}/dense_merge/bias:0'.format(i)]

        result_tensor = np.maximum(result_tensor, 0.0, result_tensor)

        # NOTE: keep in queues
        source_queues[index][offset_source] = source_tensor

        source_tensor = result_tensor

    result_tensor = np.matmul(result_tensor, variables['dense/kernel:0'])

    result_tensor += variables['dense/bias:0']

    # softmax
    result_tensor *= FLAGS.logits_scaling_factor

    # NOTE: prevent numerical error (overflow)
    result_tensor -= np.max(result_tensor)

    result_tensor = np.exp(result_tensor, result_tensor)

    result_tensor /= np.sum(result_tensor)

    return result_tensor


def decode_samples_cpu():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: read weights from the tensorflow checkpoint
    model = build_model()

    variables = {}

    with tf.Session() as session:
        # NOTE: restore the model weights
        if tf.gfile.IsDirectory(FLAGS.ckpt_path):
            ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        else:
            ckpt_path = FLAGS.ckpt_path

        tf.train.Saver().restore(session, ckpt_path)

        # NOTE: collect all kernels & bias
        trainable_variables = tf.trainable_variables()

        for v in trainable_variables:
            variables[v.name] = session.run(v)

            if len(variables[v.name].shape) == 1:
                variables[v.name] = np.reshape(variables[v.name], [1, -1])

    conditions = prepare_conditions()

    source_queues = prepare_source_queues()

    l_source_tensors = np.zeros(
        (FLAGS.num_layers + 1, FLAGS.num_quantization_levels), dtype=np.float32)

    l_condition_tensors = np.zeros(
        (FLAGS.num_layers + 1, FLAGS.condition_size), dtype=np.float32)

    # NOTE: 0 ~ 255
    decoded_samples = [128]

    # NOTE: log performance
    time_begin = time.time()

    # NOTE: skip padded zeros
    t = 2 ** FLAGS.num_layers

    while t < len(conditions):
        if len(decoded_samples) % 500 == 0:
            print('decoding samples[{}]'.format(len(decoded_samples)))

        # NOTE: previous decoded sample
        source_tensor = np.zeros(
            (1, FLAGS.num_quantization_levels), dtype=np.float32)

        source_tensor[0, decoded_samples[-1]] = 1.0

        result_tensor = numpy_fftnet(
            variables,
            source_tensor,
            source_queues,
            conditions,
            t)

        # NOTE: sample the decoded sample
        m = np.random.choice(
            np.arange(FLAGS.num_quantization_levels),
            p=result_tensor[0])

        decoded_samples.append(m)

        # NOTE: advance on timeline
        t = t + 1

    performance = (time.time() - time_begin) * 16_000 / len(decoded_samples)

    print('numpy')
    print('{}'.format(FLAGS.source_wave_path))
    print('{} seconds / 16000 samples'.format(performance))

    return decoded_samples


def decode_samples_gpu():
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
    decoded_samples = [128]

    with tf.Session() as session:
        # NOTE: restore the model weights
        if tf.gfile.IsDirectory(FLAGS.ckpt_path):
            ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        else:
            ckpt_path = FLAGS.ckpt_path

        tf.train.Saver().restore(session, ckpt_path)

        # NOTE: log performance
        time_begin = time.time()

        # NOTE: skip padded zeros
        t = 2 ** FLAGS.num_layers

        while t < len(conditions):
            if len(decoded_samples) % 500 == 0:
                print('decoding samples[{}]'.format(len(decoded_samples)))

            # NOTE: previous decoded sample
            current_source_tensor = np.zeros(
                (1, FLAGS.num_quantization_levels), dtype=np.float32)

            current_source_tensor[0, decoded_samples[-1]] = 1.0

            # NOTE: next condition for decoding next sample
            current_condition_tensor = conditions[t:t+1]

            # NOTE: build source & condition tensors
            for i in range(FLAGS.num_layers, -1, -1):
                index = FLAGS.num_layers - i
                shift = t % source_queues[index].shape[0]

                l_source_tensors[index] = source_queues[index][shift]
                l_condition_tensors[index] = conditions[t - 2 ** i]

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

            # NOTE: sample the decoded sample
            m = np.random.choice(
                np.arange(FLAGS.num_quantization_levels),
                p=fetched['result_tensor'][0])

            decoded_samples.append(m)

            # NOTE: push newly decoded intermediate data into queues
            for i in range(FLAGS.num_layers, -1, -1):
                shift = t % source_queues[i].shape[0]

                source_queues[i][shift] = fetched['r_source_tensors'][i]

            # NOTE: advance on timeline
            t = t + 1

    performance = (time.time() - time_begin) * 16_000 / len(decoded_samples)

    print('tensorflow')
    print('{}'.format(FLAGS.source_wave_path))
    print('{} seconds / 16000 samples'.format(performance))

    return decoded_samples


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.use_cpu:
        decoded_samples = decode_samples_cpu()
    else:
        decoded_samples = decode_samples_gpu()

    # NOTE: decode decoded samples
    wave = np.array(decoded_samples, dtype=np.float32)

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

    # NOTE: FFTNET, 2.3.2,
    #       For unvoiced sounds, we randomly sample from the posterior
    #       distribution; and for voiced sounds, we take the normalized logits
    #       (the input values before softmax), multiply it by a constant c > 1
    #       and pass it through the softmax layer to obtain a posterior
    #       distribution where random sampling is performed. In this way, the
    #       posterior distribution will look steeper while the original noise
    #       distribution is preserved. In this work, we use c=2.
    tf.app.flags.DEFINE_float(
        'logits_scaling_factor',
        2.0,
        'scaling factor for logits before the softmax layer')

    tf.app.flags.DEFINE_boolean(
        'use_cpu',
        False,
        'use numpy instead od tensorflow to decode the wave')

    tf.app.run()