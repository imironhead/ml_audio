"""
"""
import numpy as np
import tensorflow as tf


def fft_layer(
        offset,
        source_tensors,
        condition_tensors,
        hidden_layer_size,
        scope_name):
    """
    """
    initializer = tf.contrib.layers.xavier_initializer()

    # NOTE: FFTNET, 2.3.1, zero padding,
    #       we recommend using training sequences of length between 2N and 3N
    #       so that significant number (33% - 50%) of training samples are
    #       partial sequences.
    source_tensors = tf.pad(source_tensors, [[0, 0], [offset, 0], [0, 0]])

    l_source_tensors = source_tensors[:, :-offset, :]
    r_source_tensors = source_tensors[:, offset:, :]

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        l_tensors = tf.layers.dense(
            l_source_tensors,
            units=hidden_layer_size,
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='dense_l_samples')

        r_tensors = tf.layers.dense(
            r_source_tensors,
            units=hidden_layer_size,
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='dense_r_samples')

        if condition_tensors is not None:
            # NOTE: FFTNET, 2.3.1, zero padding,
            #       we recommend using training sequences of length between 2N
            #       and 3N so that significant number (33% - 50%) of training
            #       samples are partial sequences.
            condition_tensors = tf.pad(
                condition_tensors, [[0, 0], [offset, 0], [0, 0]])

            l_condition_tensors = condition_tensors[:, :-offset, :]
            r_condition_tensors = condition_tensors[:, offset:, :]

            l_tensors = l_tensors + tf.layers.dense(
                l_condition_tensors,
                units=hidden_layer_size,
                activation=None,
                use_bias=False,
                kernel_initializer=initializer,
                name='dense_l_conditions')

            r_tensors = r_tensors + tf.layers.dense(
                r_condition_tensors,
                units=hidden_layer_size,
                activation=None,
                use_bias=False,
                kernel_initializer=initializer,
                name='dense_r_conditions')

        tensors = tf.nn.relu(l_tensors + r_tensors)

        tensors = tf.layers.dense(
            tensors,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            name='dense_merge')

    return tensors


def light_weight_fft_layer(
        l_source_tensor,
        r_source_tensor,
        l_condition_tensor,
        r_condition_tensor,
        hidden_layer_size,
        scope_name):
    """
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        l_tensor = tf.layers.dense(
            l_source_tensor,
            units=hidden_layer_size,
            activation=None,
            use_bias=True,
            name='dense_l_samples')

        r_tensor = tf.layers.dense(
            r_source_tensor,
            units=hidden_layer_size,
            activation=None,
            use_bias=False,
            name='dense_r_samples')

        if r_condition_tensor is not None:
            l_tensor = l_tensor + tf.layers.dense(
                l_condition_tensor,
                units=hidden_layer_size,
                activation=None,
                use_bias=False,
                name='dense_l_conditions')

            r_tensor = r_tensor + tf.layers.dense(
                r_condition_tensor,
                units=hidden_layer_size,
                activation=None,
                use_bias=False,
                name='dense_r_conditions')

        tensor = tf.nn.relu(l_tensor + r_tensor)

        tensor = tf.layers.dense(
            tensor,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            name='dense_merge')

    return tensor


def build_training_model(
        num_samples,
        num_quantization_levels=256,
        hidden_layer_size=256,
        condition_size=20,
        num_layers=11):
    """
    FFTNet: a Real-Time Speaker-Dependent Neural Vocoder

    2.3.1 zero padding
    we recommend using training sequences of length between 2N and 3N so that a
    significant number (33% ~ 50%) of training samples are partial sequences.

    3.1
    in each training step, a minibatch of 5x5000 sample sequences are fed to
    the network.

    samples_size: size of input samples, we expect it 5000 + 2048

    3.1
    the fftnet implementation contains 11 fft-layers with 256 channels, which
    also has a receptive field of 2048.

    num_layers: num of fftnet layers, we expect it to be 11
    """
    source_waves = tf.placeholder(
        shape=[None, num_samples, num_quantization_levels], dtype=tf.float32)

    target_waves = tf.placeholder(
        shape=[None, num_samples, num_quantization_levels], dtype=tf.float32)

    condition_tensors = tf.placeholder(
        shape=[None, num_samples, condition_size], dtype=tf.float32)

    tensors = source_waves

    for i in range(num_layers + 1):
        tensors = fft_layer(
            2 ** (num_layers - i),
            tensors,
            condition_tensors,
            hidden_layer_size,
            'fft_{}'.format(i))

    initializer = tf.contrib.layers.xavier_initializer()

    tensors = tf.layers.dense(
        tensors,
        units=num_quantization_levels,
        activation=None,
        kernel_initializer=initializer)

    temp_source_waves = tf.reshape(tensors, [-1, num_quantization_levels])
    temp_target_waves = tf.reshape(target_waves, [-1, num_quantization_levels])

    loss = tf.losses.softmax_cross_entropy(
        temp_target_waves,
        temp_source_waves,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    step = tf.train.get_or_create_global_step()

    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.001, dtype=tf.float32),
        dtype=tf.float32)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    return {
        'source_waves': source_waves,
        'condition_tensors': condition_tensors,
        'step': step,
        'loss': loss,
        'trainer': trainer,
        'target_waves': target_waves,
        'learning_rate': learning_rate,
    }


def build_decoding_model(
        num_quantization_levels=256,
        hidden_layer_size=256,
        condition_size=20,
        num_layers=11,
        logits_scaling_factor=2.0):
    """
    """
    model = {}

    r_source_tensor = tf.placeholder(
        shape=[1, num_quantization_levels], dtype=tf.float32)

    r_condition_tensor = tf.placeholder(
        shape=[1, condition_size], dtype=tf.float32)

    model['r_condition_tensor_0'] = r_condition_tensor

    for i in range(num_layers + 1):
        if i == 0:
            l_source_tensor = tf.placeholder(
                shape=[1, num_quantization_levels], dtype=tf.float32)
        else:
            l_source_tensor = tf.placeholder(
                shape=[1, hidden_layer_size], dtype=tf.float32)

        l_condition_tensor = tf.placeholder(
            shape=[1, condition_size], dtype=tf.float32)

        model['r_source_tensor_{}'.format(i)] = r_source_tensor
        model['l_source_tensor_{}'.format(i)] = l_source_tensor
        model['l_condition_tensor_{}'.format(i)] = l_condition_tensor

        r_source_tensor = light_weight_fft_layer(
            l_source_tensor,
            r_source_tensor,
            l_condition_tensor,
            r_condition_tensor,
            hidden_layer_size,
            'fft_{}'.format(i))

    r_source_tensor = tf.layers.dense(
        r_source_tensor,
        units=num_quantization_levels,
        activation=None)

    model['result_tensor'] = \
        tf.nn.softmax(logits_scaling_factor * r_source_tensor)

    return model

