"""
"""
import numpy as np
import os
import scipy.io.wavfile
import tensorflow as tf

import datasets
import model_wavernn


def build_training_model(
        data_dir_path, state_size, batch_size, num_step_samples):
    """
    """
    # NOTE: build a training data generator which read wave samples from wav
    #       files in data_dir_path
    data_iterator = datasets.build_waves_batch_iterator(
        data_dir_path, batch_size, num_step_samples)

    next_batch = data_iterator.get_next()

    model = model_wavernn.build_model(next_batch, state_size)

    model['data_iterator'] = data_iterator

    return model


def build_generating_model(state_size):
    """
    """
    return model_wavernn.build_generative_model(state_size)


def train(session, model, initial_state, saver):
    """
    """
    session.run(model['data_iterator'].initializer)

    feeds = {
        model['initial_state']: initial_state,
        model['learning_rate']: 1e-5,
    }

    fetch = {
        'trainer': model['trainer'],
        'loss': model['loss'],
        'step': model['step'],
    }

    for i in range(1000000):
        fetched = session.run(fetch, feed_dict=feeds)

        if fetched['step'] % 1000 == 0:
            print('loss[{}]: {}'.format(fetched['step'], fetched['loss']))


def generate(session, model, initial_state, out_wave_path):
    """
    """
    samples = []

    state = initial_state

    source_waves = np.zeros((1, 512))

    idx_old_f = 0
    idx_old_c = 0

    # NOTE: 5 seconds
    for i in range(24000 * 1):
        feeds = {
            model['initial_state']: state,
            model['source_waves']: source_waves,
        }

        fetch = {
            'state': model['state'],
            'result_waves': model['result_waves'],
        }

        fetched = session.run(fetch, feed_dict=feeds)

        state = fetched['state']

        result_waves = fetched['result_waves']

        idx_new_f = np.argmax(result_waves[0, :256])
        idx_new_c = np.argmax(result_waves[0, 256:])

        new_sample = int(idx_old_c * 256 + idx_new_f) - 32768
        new_sample = min(32767, max(-32767, new_sample))

        samples.append(new_sample)

        idx_old_f = idx_new_f
        idx_old_c = idx_new_c

        source_waves = np.zeros((1, 512))

        source_waves[0, idx_new_f] = 1.0
        source_waves[0, idx_new_c + 256] = 1.0

        if i % 1000 == 0:
            print('[{}]: {}, {}'.format(i, idx_old_c, idx_old_f))
            print('[state]: {}'.format(state[0, :5]))

    samples = np.array(samples, dtype=np.int16)

    scipy.io.wavfile.write(out_wave_path, 24000, samples)


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.generate:
        model = build_generating_model(FLAGS.state_size)

        initial_state = np.zeros((1, FLAGS.state_size))
        initial_state = np.random.random((1, FLAGS.state_size))
    else:
        model = build_training_model(
            FLAGS.training_data_path,
            FLAGS.state_size,
            FLAGS.batch_size,
            FLAGS.num_step_samples)

        initial_state = np.zeros((FLAGS.batch_size, FLAGS.state_size))

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        if FLAGS.generate:
            generate(session, model, initial_state, FLAGS.out_wave_path)
        else:
            train(session, model, initial_state, saver)

            saver.save(session, target_ckpt_path, global_step=10000)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('training_data_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')

    tf.app.flags.DEFINE_string('out_wave_path', None, '')

    tf.app.flags.DEFINE_integer('state_size', 896, '')

    tf.app.flags.DEFINE_integer('batch_size', 128, '')
    tf.app.flags.DEFINE_integer('num_step_samples', 32, '')

    tf.app.flags.DEFINE_boolean('generate', False, '')

    tf.app.run()