"""
"""
import tensorflow as tf


def phase_shuffle(tensor, n=0):
    """
    arXiv:1802.04208v1, figure 5

    tensor: [batch_size, size, num_channels]
    """
    if n > 0:
        shape = tf.shape(tensor)

        # NOTE: pad n in both side (except batch & channel)
        paddings = tf.constant([[0, 0], [n, n], [0, 0]])

        tensor = tf.pad(
            tensor,
            paddings=paddings,
            mode='REFLECT')

        # NOTE: random crop back to original size
        tensor = tf.random_crop(tensor, shape)

    return tensor


def build_discriminator(
        tensor,
        num_channels,
        model_size,
        shuffle_phase,
        scope_name):
    """
    arXiv:1802.04208v1, table 4

    to discriminate if a wave segment is real

    tensor: wave samples (n, 16384, c)
    num_channels: num of channels of generated audio wave
    model_size: d in the table
    scope_name: scope name of this sub-network
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        for out_dim in [1, 2, 4, 8, 16]:
            tensor = tf.layers.conv1d(
                tensor,
                filters=out_dim * model_size,
                kernel_size=25,
                strides=4,
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=initializer)

            if shuffle_phase and out_dim < 16:
                tensor = phase_shuffle(tensor, 2)

        tensor = tf.layers.flatten(tensor)

        tensor = tf.layers.dense(
            tensor,
            units=1,
            activation=None,
            kernel_initializer=initializer)

    return tensor


def build_generator(tensor, num_channels, model_size, scope_name):
    """
    arXiv:1802.04208v1, table 3

    to generate a wave segment (16384 samples, ~1 second of 16k audio)

    tensor: a random uniform vector (n, 100)
    num_channels: num of channels of generated audio wave
    model_size: d in the table
    scope_name: scope name of this sub-network
    """
    N = tf.shape(tensor)[0]

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        tensor = tf.layers.dense(
            tensor,
            units=256 * model_size,
            activation=tf.nn.relu,
            kernel_initializer=initializer)

        tensor = tf.reshape(tensor, [-1, 16, 16 * model_size])

        layers = [
            {'len':    64, 'src': 16 * model_size, 'tar': 8 * model_size, 'activation': tf.nn.relu},
            {'len':   256, 'src':  8 * model_size, 'tar': 4 * model_size, 'activation': tf.nn.relu},
            {'len':  1024, 'src':  4 * model_size, 'tar': 2 * model_size, 'activation': tf.nn.relu},
            {'len':  4096, 'src':  2 * model_size, 'tar': 1 * model_size, 'activation': tf.nn.relu},
            {'len': 16384, 'src':  1 * model_size, 'tar': num_channels, 'activation': tf.nn.tanh},
        ]

        for layer in layers:
            # NOTE: nooooo .... conv1d_transpose
            kernel_size_source = layer['src']
            kernel_size_target = layer['tar']

            kernel = tf.get_variable(
                name='conv1d_transpose_{}'.format(layer['src']),
                shape=[25, kernel_size_target, kernel_size_source],
                dtype=tf.float32,
                initializer=initializer)

            tensor = tf.contrib.nn.conv1d_transpose(
                tensor,
                filter=kernel,
                output_shape=[N, layer['len'], kernel_size_target],
                stride=4,
                padding='SAME',
                data_format='NWC')

            tensor = layer['activation'](tensor)

    return tensor


def build_model(seed, real, num_channels, model_size, shuffle_phase):
    """
    arXiv:1704.00028
    Improved training of Wasserstein GANs

    seed: random tensors as input of the generator
    real: tensors of real wave. None if discriminator is not needed.
    """
    model = {}

    # NOTE: build the generator to generate images from random seeds.
    fake = build_generator(seed, num_channels, model_size, 'g_')

    model['seed'] = seed
    model['fake'] = fake

    # NOTE: build training model if real tensors is not None
    if real is None:
        return model

    step = tf.train.get_or_create_global_step()

    # NOTE: build the discriminator to judge the real data.
    d_real = build_discriminator(
        real, num_channels, model_size, shuffle_phase, 'd_')

    # NOTE: build the discriminator to judge the fake data.
    #       judge both real and fake data with the same network (shared).
    d_fake = build_discriminator(
        fake, num_channels, model_size, shuffle_phase, 'd_')

    # NOTE: gradient penalty
    alpha = tf.random_uniform([tf.shape(seed)[0], 1, 1])
    inter = fake + alpha * (real - fake)

    d_inte = build_discriminator(
        inter, num_channels, model_size, shuffle_phase, 'd_')

    gradients = tf.gradients(d_inte, inter)[0]

    gradients_norm = tf.norm(gradients, 2, axis=1)

    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2.0)

    # NOTE: loss
    d_loss = tf.reduce_mean(d_fake - d_real) + 1.0 * gradient_penalty

    g_loss = -tf.reduce_mean(d_fake)

    # NOTE: collect variables to separate g/d training op
    d_variables = []
    g_variables = []

    for variable in tf.trainable_variables():
        if variable.name.startswith('d_'):
            d_variables.append(variable)
        elif variable.name.startswith('g_'):
            g_variables.append(variable)

    g_trainer = \
        tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
    g_trainer = g_trainer.minimize(
        g_loss,
        global_step=step,
        var_list=g_variables,
        colocate_gradients_with_ops=True)

    d_trainer = \
        tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
    d_trainer = d_trainer.minimize(
        d_loss,
        var_list=d_variables,
        colocate_gradients_with_ops=True)

    model['step'] = step
    model['real'] = real
    model['g_loss'] = g_loss
    model['d_loss'] = d_loss
    model['g_trainer'] = g_trainer
    model['d_trainer'] = d_trainer

    return model

