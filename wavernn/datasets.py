"""
"""
import numpy as np
import os
import scipy.io.wavfile
import tensorflow as tf


def random_waves(dir_path, num_step_samples):
    """
    """
    names = os.listdir(dir_path)
    names = [name for name in names if name.endswith('.wav')]
    paths = [os.path.join(dir_path, name) for name in names]

    while True:
        np.random.shuffle(paths)

        for path in paths:
            # NOTE: read wave, assume is 16bit in 24,000 Hz
            _, data = scipy.io.wavfile.read(path)

            # NOTE: drop residual
            num_samples = data.shape[0]

            if num_samples % num_step_samples == 0:
                window_len = num_samples + 1 - num_step_samples
            else:
                window_len = num_samples + 1 - num_samples % num_step_samples

            window_pos = np.random.randint(
                num_samples - window_len + 1)

            data = data[window_pos:window_pos+window_len]

            # NOTE: int16 to uint16
            data = data + 32768

            # NOTE: to coarse and fine
            data_coarse = data // 256
            data_fine = data % 256

            # NOTE: shift and merge (c(t-1), f(t-1), c(t))
            data = np.stack(
                [data_coarse[:-1], data_fine[:-1], data_coarse[1:]], axis=1)

            yield data


def build_waves_generator(dir_path, num_step_samples):
    """
    """
    def waves_generator():
        """
        """
        waves = random_waves(dir_path, num_step_samples)

        # NOTE: to index one sample (do onehot encoding)
        idxs = np.arange(num_step_samples)

        sample_sources = []

        while True:
            # NOTE: prepare source, keep 4 different sources to mix samples
            while len(sample_sources) < 4:
                sample_sources.append(next(waves))

            # NOTE: yield one training sample
            source = sample_sources.pop(0)

            sample = source[:num_step_samples, :]

            # NOTE: push source back if it still has data
            if source.shape[0] > num_step_samples:
                sample_sources.append(source[num_step_samples:, :])

            # NOTE: make onehot version
            onehot = np.zeros((num_step_samples, 768))

            onehot[idxs, sample[idxs, 0]] = 1.0
            onehot[idxs, sample[idxs, 1] + 256] = 1.0
            onehot[idxs, sample[idxs, 2] + 512] = 1.0

            yield onehot

    return waves_generator


def build_waves_batch_iterator(dir_path, batch_size=128, num_step_samples=32):
    """
    """
    # NOTE: build clip generator with waves inside dir_path
    clips = build_waves_generator(dir_path, num_step_samples)

    data = tf.data.Dataset.from_generator(
        clips,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([num_step_samples, 768]))

    # NOTE: a pool to shuffle samples
    data = data.shuffle(buffer_size=512)

    # NOTE: combine clips to batches
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator

