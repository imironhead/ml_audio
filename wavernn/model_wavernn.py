"""
"""
import numpy as np
import tensorflow as tf


def build_cell_function(state_size):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.2)

    def dense(h, x, r, name):
        a = tf.layers.dense(
            h,
            units=state_size//2,
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='{}_a'.format(name))

        b = tf.layers.dense(
            x,
            units=state_size//2,
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='{}_b'.format(name))

        if r is not None:
            a = a * r

        return a + b

    def predict(state_p, name):
        p = tf.layers.dense(
            state_p,
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            name='{}_p_1024'.format(name))

        p = tf.layers.dense(
            p,
            units=256,
            activation=None,
            kernel_initializer=initializer,
            name='{}_p_256'.format(name))

        return p

    def cell(x, state):
        """
        arXiv:1802.08435v1 - equation 2

        x: [N, 256 + 256]
        """
        state_c, state_f = tf.split(state, 2, axis=-1)

        # NOTE: predict coarse part
        x_c = tf.pad(x, paddings=[[0, 0], [0, 256]])

        with tf.variable_scope('wavernn_cell', reuse=tf.AUTO_REUSE):
            u_c = tf.nn.sigmoid(dense(state, x_c, None, 'u_u'))
            r_c = tf.nn.sigmoid(dense(state, x_c, None, 'r_u'))
            e_c = tf.nn.tanh(dense(state, x_c, r_c, 'e_u'))

            state_c = state_c * u_c - e_c * (u_c - 1.0)

            p_c = predict(state_c, 'predict_c')

            # NOTE: predict coarse part
            p_t = tf.nn.softmax(p_c)

            x_f = tf.concat([x, p_t], axis=-1)

            u_f = tf.nn.sigmoid(dense(state, x_f, None, 'u_l'))
            r_f = tf.nn.sigmoid(dense(state, x_f, None, 'r_l'))
            e_f = tf.nn.tanh(dense(state, x_f, r_f, 'e_l'))

            state_f = state_f * u_f - e_f * (u_f - 1.0)

            p_f = predict(state_f, 'predict_f')

        #
        state = tf.concat([state_c, state_f], axis=-1)

        return p_c, p_f, state

    return cell


def build_model(step_size, state_size, training):
    """
    """
    # NOTE: sanity check, one step for generation
    if not training:
        step_size = 1

    source_waves = tf.placeholder(
        shape=[None, step_size, 512], dtype=tf.float32)

    # NOTE: tensors shape should be [N, T, D]
    #       N: batch size
    #       T: time step
    #       D: 256 (fine) + 256 (coarse)
    xs = tf.unstack(source_waves, axis=1)

    # NOTE: build a function to build rnn cells
    cell_function = build_cell_function(state_size=state_size)

    # NOTE: initial state
    state = initial_state = tf.placeholder(
        shape=[None, state_size], dtype=tf.float32)

    # NOTE: collect result samples
    c_results = []
    f_results = []

    # NOTE: do recurrence
    for x in xs:
        pc_guess, pf_guess, state = cell_function(x, state)

        c_results.append(pc_guess)
        f_results.append(pf_guess)

    result_waves_c = tf.stack(c_results, axis=1)
    result_waves_f = tf.stack(f_results, axis=1)

    model = {
        'head_state': initial_state,
        'tail_state': state,
        'source_waves': source_waves,
    }

    if not training:
        c_indices = tf.argmax(result_waves_c, axis=-1)
        f_indices = tf.argmax(result_waves_f, axis=-1)

        c_onehot = tf.one_hot(c_indices, 256)
        f_onehot = tf.one_hot(f_indices, 256)

        result_waves = tf.concat([c_onehot, f_onehot], axis=2)

        model['result_waves'] = result_waves

        return model

    # NOTE: build training part
    target_waves = tf.placeholder(
        shape=[None, step_size, 512], dtype=tf.float32)

    target_waves_c, target_waves_f = tf.split(target_waves, 2, axis=-1)

    loss = \
        tf.losses.softmax_cross_entropy(target_waves_c, result_waves_c) + \
        tf.losses.softmax_cross_entropy(target_waves_f, result_waves_f)

    # NOTE: trainer
    step = tf.train.get_or_create_global_step()

    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(1e-5, dtype=tf.float32),
        dtype=tf.float32)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    model['target_waves'] = target_waves
    model['learning_rate'] = learning_rate
    model['trainer'] = trainer
    model['loss'] = loss
    model['step'] = step

    return model


def build_generative_model(state_size):
    """
    """
    # NOTE: source_waves is a batch of one samples, the shape should be [N, D]
    #       N: batch size
    #       D: 256 (coarse) + 256 (fine)
    source_waves = tf.placeholder(shape=[None, 512], dtype=tf.float32)

    # NOTE: pad 0 as placeholder of c(t) which is masked for predicting c(t)
#   tensors = tf.pad(source_waves, [[0, 0], [0, 256]])

    # NOTE: build a function to build rnn cells
    cell_function = build_cell_function(state_size=state_size)

    # NOTE: initial state
    initial_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

    # NOTE: use fine(t-1) & coarse(t-1) to infer coarse(t)

    pc_guess, pf_guess, state = cell_function(source_waves, initial_state)

    c_indices = tf.argmax(pc_guess, axis=-1)
    f_indices = tf.argmax(pf_guess, axis=-1)

    c_onehot = tf.one_hot(c_indices, 256)
    f_onehot = tf.one_hot(f_indices, 256)

    result_waves = tf.concat([c_onehot, f_onehot], axis=1)

    return {
        'initial_state': initial_state,
        'state': state,
        'source_waves': source_waves,
        'result_waves': result_waves,
    }

