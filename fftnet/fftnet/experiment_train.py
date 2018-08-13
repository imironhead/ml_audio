"""
"""
import os

import numpy as np
import tensorflow as tf

import fftnet.dataset as dataset
import fftnet.model_fftnet as model_fftnet


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = model_fftnet.build_training_model(
        num_samples=FLAGS.samples_size,
        num_quantization_levels=FLAGS.num_quantization_levels,
        condition_size=FLAGS.condition_size,
        num_layers=FLAGS.num_layers)

    return model


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    target_size = FLAGS.samples_size

    batches = dataset.wave_batches(
        FLAGS.train_dir_path, FLAGS.batchs_size, FLAGS.samples_size)

    for clean_waves, noise_waves, features in batches:
        # NOTE: use noisy input to infer clean output
        sources = noise_waves
        targets = clean_waves
        conditions = features / FLAGS.condition_scale

        yield sources, conditions, targets


def build_summaries(model):
    """
    """
    return {'loss': tf.summary.scalar('loss', model['loss'])}


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

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

        # NOTE: give up overlapped old data
        step = session.run(model['step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        while True:
            step = session.run(model['step'])

            learning_rate = FLAGS.learning_rate * (0.5 ** (step // 100000))

#           learning_rate = FLAGS.learning_rate

            if step > 60000 and step % 20000 == 0:
                saver.save(session, target_ckpt_path, global_step=step)

            source_waves, conditions, target_waves = next(training_batches)

            fetch = {
                'step': model['step'],
                'trainer': model['trainer'],
                'summary': summaries['loss'],
            }

            feeds = {
                model['source_waves']: source_waves,
                model['target_waves']: target_waves,
                model['condition_tensors']: conditions,
                model['learning_rate']: learning_rate,
            }

            fetched = session.run(fetch, feed_dict=feeds)

            reporter.add_summary(fetched['summary'], fetched['step'])


if __name__ == '__main__':
    tf.app.flags.DEFINE_float('learning_rate', 0.001, '')

    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')

    tf.app.flags.DEFINE_integer('batchs_size', 5, '')
    tf.app.flags.DEFINE_integer('samples_size', 5000, '')

    # NOTE: FFTNET, 3.1, experimental setup,
    #       the waveforms are quantized to 256 categorical values based on
    #       mu-law.
    tf.app.flags.DEFINE_integer('num_quantization_levels', 256, '')

    # NOTE: FFTNET, 2.2
    #       Then we extract the MCC and F0 features for each overlapping
    #       window. For the ht corresponding to the window centers, we assign
    #       the computed MCC and F0 values (26 dimensions in total).
    # NOTE: I do not know how to get F0, use mfcc instead.
    tf.app.flags.DEFINE_integer('condition_size', 32, '')

    tf.app.flags.DEFINE_float('condition_scale', 1000.0, '')

    # NOTE: FFTNET, 3.1
    #       The FFTNet implementation contains 11 FFT-layers with 256 channels,
    #       which also has a receptive field of 2048.
    tf.app.flags.DEFINE_integer('num_layers', 11, '')

    tf.app.run()

