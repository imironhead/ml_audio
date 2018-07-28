"""
"""
import tensorflow as tf


def fft_layer(l_tensors, r_tensors, scope_name):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        l_tensors = tf.layers.dense(
            l_tensors,
            units=256,
            activation=None,
            kernel_initializer=initializer)

        r_tensors = tf.layers.dense(
            r_tensors,
            units=256,
            activation=None,
            use_bias=False,
            kernel_initializer=initializer)

        tensors = tf.nn.relu(l_tensors + r_tensors)

        tensors = tf.layers.dense(
            tensors,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=initializer)

    return tensors


def build_training_model(samples_size, num_layers):
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
    target_size = samples_size - 2 ** num_layers

    # TODO: onehot or 0 ~ 256 or -1.0 ~ +1.0, which one is better?
    # NOTE: current -1.0 ~ +1.0
    source_waves = tf.placeholder(
        shape=[None, samples_size - 1, 1], dtype=tf.float32)

    target_waves = tf.placeholder(
        shape=[None, target_size, 256], dtype=tf.float32)

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = source_waves

    for i in range(num_layers - 1, -1, -1):
        offset_size = 2 ** i

        l_tensors = tensors[:, :-offset_size, :]
        r_tensors = tensors[:, offset_size:, :]

        tensors = fft_layer(l_tensors, r_tensors, 'fft_{}'.format(i))

    tensors = tf.layers.dense(
        tensors,
        units=256,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer)

    loss = tf.losses.softmax_cross_entropy(
        target_waves, tensors, reduction=tf.losses.Reduction.MEAN)

    step = tf.train.get_or_create_global_step()

    trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0001) \
        .minimize(loss, global_step=step)

    return {
        'step': step,
        'loss': loss,
        'trainer': trainer,
        'source_waves': source_waves,
        'target_waves': target_waves,
    }


def build_generative_model():
    """
    """


def build_model(training, samples_size, num_layers):
    """
    """
    if training:
        model = build_training_model(samples_size, num_layers)
    else:
        model = build_generative_model()

    return model

