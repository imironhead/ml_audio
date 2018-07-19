"""
"""
import os

import tensorflow as tf

import wavegan.model_wavegan as model_wavegan


def build_training_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: sanity check, the paper try to generate waves with 16384 samples
    FLAGS.num_target_samples = 16384
    FLAGS.num_source_samples = min(16384, max(0, FLAGS.num_source_samples))

    if FLAGS.use_speech_commands:
        import wavegan.dataset_speech_commands as dataset

        data_iterator = dataset.build_waves_batch_iterator(
            FLAGS.train_dir_path,
            FLAGS.use_sc09,
            FLAGS.num_source_samples,
            FLAGS.num_target_samples,
            FLAGS.batch_size)
    else:
        import wavegan.dataset_waves as dataset

        data_iterator = dataset.build_waves_batch_iterator(
            FLAGS.train_dir_path,
            FLAGS.num_target_samples,
            FLAGS.batch_size)

    waves = data_iterator.get_next()

    # NOTE: random vector for the generator
    seeds = tf.random_uniform(
        shape=[FLAGS.batch_size, FLAGS.seed_size], minval=-1.0, maxval=1.0)

    model = model_wavegan.build_model(
        seeds,
        waves,
        FLAGS.num_channels,
        FLAGS.model_size,
        FLAGS.gradient_penalty_lambda,
        FLAGS.shuffle_phase)

    model['data_iterator'] = data_iterator

    return model


def build_summaries(model):
    """
    """
    summary_audio = tf.summary.audio('audio', model['fakes'][:2], 16000)
    summary_d_loss = tf.summary.scalar('discriminator_loss', model['d_loss'])
    summary_g_loss = tf.summary.scalar('generator_loss', model['g_loss'])

    return {
        'summary_audio': summary_audio,
        'summary_d_loss': summary_d_loss,
        'summary_g_loss': summary_g_loss,
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_training_model()

    reporter = tf.summary.FileWriter(FLAGS.log_path)

    summaries = build_summaries(model)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        # NOTE: initialize the data iterator
        session.run(model['data_iterator'].initializer)

        while True:
            step = session.run(model['step'])

            if (step + 1) % 10000 == 0:
                saver.save(session, target_ckpt_path, global_step=step)

            # NOTE: train discriminator
            fetch = {
                'loss': model['d_loss'],
                'step': model['step'],
                'trainer': model['d_trainer'],
                'summary': summaries['summary_d_loss'],
            }

            fetched = session.run(fetch)

            reporter.add_summary(fetched['summary'], fetched['step'])

            # NOTE: train generator
            fetch = {
                'loss': model['g_loss'],
                'step': model['step'],
                'trainer': model['g_trainer'],
                'summary': summaries['summary_g_loss'],
            }

            if step % 1000 == 0:
                fetch['audio'] = summaries['summary_audio']

            fetched = session.run(fetch)

            reporter.add_summary(fetched['summary'], fetched['step'])

            if 'audio' in fetched:
                reporter.add_summary(fetched['audio'], fetched['step'])

            if step % 100 == 0:
                print(step)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')

    tf.app.flags.DEFINE_float('gradient_penalty_lambda', 1.0, '')

    tf.app.flags.DEFINE_integer('seed_size', 100, '')
    tf.app.flags.DEFINE_integer('model_size', 64, '')
    tf.app.flags.DEFINE_integer('batch_size', 64, '')
    tf.app.flags.DEFINE_integer('num_channels', 1, '')
    tf.app.flags.DEFINE_integer('num_source_samples', 15000, '')
    tf.app.flags.DEFINE_integer('num_target_samples', 16384, '')

    tf.app.flags.DEFINE_boolean('shuffle_phase', False, '')
    tf.app.flags.DEFINE_boolean('use_speech_commands', True, '')
    tf.app.flags.DEFINE_boolean('use_sc09', False, '')

    tf.app.run()

