"""
"""
import os

import numpy as np
import tensorflow as tf

import fftnet.dataset as dataset
import fftnet.model_fftnet as model_fftnet


def sanity_check():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    FLAGS.zeros_size = 2 ** FLAGS.num_layers


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = model_fftnet.build_model(
        True,
        FLAGS.samples_size + 2 ** FLAGS.num_layers,
        FLAGS.num_layers)

    return model


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    target_size = FLAGS.samples_size

    batches = dataset.wave_batches(
        FLAGS.train_dir_path,
        FLAGS.batchs_size,
        FLAGS.samples_size,
        FLAGS.zeros_size)

    for sources in batches:
        targets = []

        for source in sources:
            target = np.zeros((1, target_size, 256), dtype=np.float32)

            # FIXME: better method?
            for i in range(target_size):
                target[0, i, source[0, FLAGS.zeros_size + i, 0]] = 1.0

            targets.append(target)

        sources = np.concatenate(sources, axis=0)
        targets = np.concatenate(targets, axis=0)

        sources = sources[:, :-1, :]

        yield sources, targets


def build_summaries(model):
    """
    """
    return {'loss': tf.summary.scalar('loss', model['loss'])}


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    sanity_check()

    model = build_model()

    summaries = build_summaries(model)

    training_batches = build_dataset()

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        while True:
            source_waves, target_waves = next(training_batches)

            fetch = {
                'step': model['step'],
                'trainer': model['trainer'],
                'summary': summaries['loss'],
            }

            feeds = {
                model['source_waves']: source_waves,
                model['target_waves']: target_waves,
            }

            fetched = session.run(fetch, feed_dict=feeds)

            reporter.add_summary(fetched['summary'], fetched['step'])


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')

    tf.app.flags.DEFINE_integer('batchs_size', 5, '')
    tf.app.flags.DEFINE_integer('samples_size', 5000, '')
    tf.app.flags.DEFINE_integer('zeros_size', 2048, '')
    tf.app.flags.DEFINE_integer('num_layers', 11, '')

    tf.app.run()

