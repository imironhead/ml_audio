"""
speech commands dataset, 16kHz, ~1 second wave

https://www.tensorflow.org/tutorials/audio_recognition
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
"""
import numpy as np
import os
import scipy.io.wavfile
import tensorflow as tf


def random_wave_paths(dir_path, sc09_subset):
    """
    dir_path: path to dir which contains all speech command categories
    sc09_subset: only use only zero ~ nine speech command categories
    """
    commands = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine']

    if not sc09_subset:
        commands.extend([
            'bird', 'dog', 'go', 'house', 'no', 'on', 'right', 'up', 'wow',
            'bed', 'cat', 'down', 'happy', 'left', 'marvin', 'off', 'sheila',
            'stop'])

    paths = []

    for command in commands:
        names = os.listdir(os.path.join(dir_path, command))
        names = filter(lambda x: x.endswith('.wav'), names)
        temps = map(lambda x: os.path.join(dir_path, command, x), names)

        paths.extend(temps)

    while True:
        np.random.shuffle(paths)

        for path in paths:
            yield path


def build_random_wave_generator(
        dir_path, sc09_subset, num_source_samples, num_target_samples):
    """
    """
    def random_waves():
        """
        dir_path: path to dir which contains all speech command categories
        sc09_subset: only use only zero ~ nine speech command categories
        num_source_samples: random crop wave to this size.
        num_target_samples: pad 0 to cropped waves to this size
        """
        for path in random_wave_paths(dir_path, sc09_subset):
            # NOTE: read wave, assume is 16bit in 16,000 Hz
            _, data = scipy.io.wavfile.read(path)

            # NOTE: to -1.0 ~ +1.0
            data = data.astype(np.float32) / 32768.0

            # NOTE: random crop
            if data.shape[0] > num_source_samples:
                b = np.random.randint(data.shape[0] - num_source_samples)
                e = b + num_source_samples

                data = data[b:e]

            # NOTE: random pad
            if data.shape[0] < num_target_samples:
                b = np.random.randint(num_target_samples - data.shape[0])
                e = b + data.shape[0]

                temp = np.zeros(num_target_samples, dtype=np.float32)

                temp[b:e] = data

                data = temp

            # NOTE: sanity check
            if data.shape[0] > num_target_samples:
                data = data[:num_target_samples]

            # NOTE: add channel dimension
            data = np.reshape(data, [-1, 1])

            yield data

    return random_waves


def build_waves_batch_iterator(
        dir_path, sc09_subset, num_source_samples, num_target_samples,
        batch_size=128):
    """
    """
    # NOTE: build wave generator with waves inside dir_path
    waves = build_random_wave_generator(
        dir_path, sc09_subset, num_source_samples, num_target_samples)

    data = tf.data.Dataset.from_generator(
        waves,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([num_target_samples, 1]))

    # NOTE: a pool to shuffle samples
    data = data.shuffle(buffer_size=512)

    # NOTE: combine waves to batches
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator

