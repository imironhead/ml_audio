"""
"""
import numpy as np
import tensorflow as tf


def build_cell_function(state_size):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.2)

    with tf.variable_scope('wavernn_cell', reuse=tf.AUTO_REUSE):
        w_r = tf.get_variable(
            'R',
            [state_size, 3 * state_size],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        w_i = tf.get_variable(
            'I',
            [768, 3 * state_size],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        b = tf.get_variable(
            'b',
            [3 * state_size],
            trainable=True,
            initializer=tf.zeros_initializer,
            dtype=tf.float32)

        o1 = tf.get_variable(
            'o1',
            [state_size // 2, 1024],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        o3 = tf.get_variable(
            'o3',
            [state_size // 2, 1024],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        o2 = tf.get_variable(
            'o2',
            [1024, 256],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        o4 = tf.get_variable(
            'o4',
            [1024, 256],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        b1 = tf.get_variable(
            'b1',
            [1024],
            trainable=True,
            initializer=tf.zeros_initializer,
            dtype=tf.float32)

        b3 = tf.get_variable(
            'b3',
            [1024],
            trainable=True,
            initializer=tf.zeros_initializer,
            dtype=tf.float32)

        b2 = tf.get_variable(
            'b2',
            [256],
            trainable=True,
            initializer=tf.zeros_initializer,
            dtype=tf.float32)

        b4 = tf.get_variable(
            'b4',
            [256],
            trainable=True,
            initializer=tf.zeros_initializer,
            dtype=tf.float32)

        h_state_size = state_size // 2

        mask = np.ones((768, 3 * state_size), dtype=np.float32)

        mask[512:, 0*state_size:0*state_size+h_state_size] = 0.0
        mask[512:, 1*state_size:1*state_size+h_state_size] = 0.0
        mask[512:, 2*state_size:2*state_size+h_state_size] = 0.0

        mask = tf.constant(mask, name='mask')

        w_i = w_i * mask

    def cell(x, state):
        """
        arXiv:1802.08435v1 - equation 2
        """
        # NOTE: predict coarse part
        tensors_r = tf.matmul(state, w_r)
        tensors_i = tf.matmul(x, w_i) + b

        tensors_ru, tensors_rr, tensors_re = tf.split(tensors_r, 3, axis=-1)
        tensors_iu, tensors_ir, tensors_ie = tf.split(tensors_i, 3, axis=-1)

        tensors_u = tf.nn.sigmoid(tensors_ru + tensors_iu)
        tensors_r = tf.nn.sigmoid(tensors_rr + tensors_ir)
        tensors_e = tf.nn.tanh(tensors_r * tensors_re + tensors_ie)

        state = tensors_u * state - (tensors_u - 1.0) * tensors_e

        yf, yc = tf.split(state, 2, axis=-1)

        pf = tf.nn.relu(tf.matmul(yf, o1) + b1)
        pf = tf.matmul(pf, o2) + b2

        pc = tf.nn.relu(tf.matmul(yc, o3) + b3)
        pc = tf.matmul(pc, o4) + b4

        return pc, pf, state

    return cell


def build_model(source_waves, state_size):
    """
    """
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

    # NOTE: collect loss from each time step
    losses = []

    # NOTE: collect result samples
    results = []

    # TODO: for generating
    for i in range(len(xs) - 1):
        # NOTE: split into fine(t) & coarse(t+1)
        pc_label, pf_label, _ = tf.split(xs[i + 1], 3, axis=1)

        # NOTE: use fine(t-1) & coarse(t) to infer fine(t) & coarse(t+1)
        pc_guess, pf_guess, state = cell_function(xs[i], state)

        # NOTE: collect result wave samples (as format of input tensors)
        results.append(tf.concat([pf_guess, pc_guess], axis=1))

        # NOTE: softmax loss for predicting fine(t)
        loss_f = tf.losses.softmax_cross_entropy(pf_label, pf_guess)

        # NOTE: softmax loss for predicting coarse(t+1)
        loss_c = tf.losses.softmax_cross_entropy(pc_label, pc_guess)

        losses.append(loss_f + loss_c)

    result_waves = tf.stack(results, axis=1)

    loss = tf.reduce_mean(losses)

    # NOTE: trainer
    step = tf.get_variable(
        'step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(1e-5, dtype=tf.float32),
        dtype=tf.float32)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    return {
        'initial_state': initial_state,
        'step': step,
        'loss': loss,
        'state': state,
        'trainer': trainer,
        'learning_rate': learning_rate,
        'source_waves': source_waves,
        'result_waves': result_waves,
    }


def build_generative_model(state_size):
    """
    """
    # NOTE: source_waves is a batch of one samples, the shape should be [N, D]
    #       N: batch size
    #       D: 256 (coarse) + 256 (fine)
    source_waves = tf.placeholder(shape=[None, 512], dtype=tf.float32)

    # NOTE: pad 0 as placeholder of c(t) which is masked for predicting c(t)
    tensors = tf.pad(source_waves, [[0, 0], [0, 256]])

    # NOTE: build a function to build rnn cells
    cell_function = build_cell_function(state_size=state_size)

    # NOTE: initial state
    initial_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

    # NOTE: use fine(t-1) & coarse(t-1) to infer coarse(t)
    pc_guess, pf_guess, temp_state = cell_function(tensors, initial_state)

    c_indices = tf.argmax(pc_guess, axis=-1)

    c_onehot = tf.one_hot(c_indices, 256)

    tensors = tf.concat([source_waves, c_onehot], axis=1)

    pc_guess, pf_guess, state = cell_function(tensors, initial_state)

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

