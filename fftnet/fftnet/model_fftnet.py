"""
"""
import tensorflow as tf


def fft_layer(
        offset,
        source_tensors,
        condition_tensors,
        hidden_size,
        scope_name):
    """
    """
    initializer = tf.contrib.layers.xavier_initializer()

    shape = tf.shape(source_tensors)

    n, w, c = shape[0], shape[1], shape[2]

    pad_source = tf.ones([n, offset, 1])

    pad_source = tf.pad(pad_source, [[0, 0], [0, 0], [128, 127]])

#   pad_source = tf.zeros([n, offset, c])

#   pad_source[:, :, 128] = 1.0

    source_tensors = tf.concat([pad_source, source_tensors], axis=1)

#   source_tensors = tf.pad(source_tensors, [[0, 0], [offset, 0], [0, 0]])

    l_source_tensors = source_tensors[:, :-offset, :]
    r_source_tensors = source_tensors[:, offset:, :]

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        l_tensors = tf.layers.dense(
            l_source_tensors,
            units=hidden_size,
            activation=None,
            kernel_initializer=initializer)

        r_tensors = tf.layers.dense(
            r_source_tensors,
            units=hidden_size,
            activation=None,
            kernel_initializer=initializer)

        if condition_tensors is not None:
            condition_tensors = condition_tensors[:, -w:, :]

            condition_tensors = tf.pad(condition_tensors, [[0, 0], [offset, 0], [0, 0]])

            l_condition_tensors = condition_tensors[:, :-offset, :]
            r_condition_tensors = condition_tensors[:, offset:, :]

            l_tensors = l_tensors + tf.layers.dense(
                l_condition_tensors,
                units=hidden_size,
                activation=None,
                kernel_initializer=initializer)

            r_tensors = r_tensors + tf.layers.dense(
                r_condition_tensors,
                units=hidden_size,
                activation=None,
                kernel_initializer=initializer)

        tensors = tf.nn.relu(l_tensors + r_tensors)

        tensors = tf.layers.dense(
            tensors,
            units=hidden_size,
            activation=tf.nn.relu,
            kernel_initializer=initializer)

    return tensors


def build_model(
        training,
        source_size,
        target_size,
        sample_size=256,
        condition_size=20,
        hidden_size=256,
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
    # NOTE: sanity check
#   if target_size + 2 ** num_layers != source_size + 1:
#       raise Exception('invalide source_size/target_size/num_layers')

    source_waves = tf.placeholder(
        shape=[None, source_size, sample_size], dtype=tf.float32)

    condition_tensors = tf.placeholder(
        shape=[None, source_size, condition_size], dtype=tf.float32)

    tensors = source_waves

    for i in range(num_layers, -1, -1):
        tensors = fft_layer(
            2 ** i,
            tensors,
            condition_tensors,
            hidden_size,
            'fft_{}'.format(i))

    initializer = tf.contrib.layers.xavier_initializer()

    tensors = tf.layers.dense(
        tensors,
        units=sample_size,
        activation=None,
#       use_bias=False,
        kernel_initializer=initializer)

    model = {
        'source_waves': source_waves,
        'condition_tensors': condition_tensors,
    }

    if not training:
        model['result_waves'] = tf.nn.softmax(2.0 * tensors)

        return model

    target_waves = tf.placeholder(
        shape=[None, target_size, sample_size], dtype=tf.float32)

#   loss = tf.losses.softmax_cross_entropy(
#       target_waves, tensors, reduction=tf.losses.Reduction.MEAN)

    temp_target_waves = tf.reshape(target_waves, [-1, sample_size])
    tensors = tf.reshape(tensors, [-1, sample_size])

    loss = tf.losses.softmax_cross_entropy(
        temp_target_waves, tensors, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

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

    model['step'] = step
    model['loss'] = loss
    model['trainer'] = trainer
    model['target_waves'] = target_waves
    model['learning_rate'] = learning_rate

    return model


def build_generative_model(
        sample_size=256,
        condition_size=20,
        hidden_size=256,
        num_layers=11):
    """
    """
    source_size = 2 ** num_layers

    # TODO: onehot or 0 ~ 256 or -1.0 ~ +1.0, which one is better?
    # NOTE: current -1.0 ~ +1.0
    source_waves = tf.placeholder(
        shape=[None, source_size, sample_size], dtype=tf.float32)

    condition_tensors = tf.placeholder(
        shape=[None, source_size, condition_size], dtype=tf.float32)

    tensors = source_waves

    for i in range(num_layers - 1, -1, -1):
        tensors = fft_layer(
            2 ** i,
            tensors,
            condition_tensors if i + 1 == num_layers else None,
            hidden_size,
            'fft_{}'.format(i))

    result_waves = tf.layers.dense(
        tensors,
        units=sample_size,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer)

    return {
        'source_waves': source_waves,
        'result_waves': result_waves,
    }



