"""
"""
import numpy as np
import os
import scipy.io.wavfile
import tensorflow as tf


def random_waves(dir_path):
    """
    """
    names = os.listdir(dir_path)
    names = filter(lambda x: x.endswith('.wav'), names)
    paths = [os.path.join(dir_path, name) for name in names]

    while True:
        np.random.shuffle(paths)

        for path in paths:
            # NOTE: read wave, assume they are 16-bits in 24,000 Hz
            _, data = scipy.io.wavfile.read(path)

            data = data + 32768

            yield data


def build_waves_batch_iterator(dir_path, batch_size=128, num_step_samples=32):
    """
    """
    # NOTE: infinit wave ganerator
    waves = random_waves(dir_path)

    # NOTE: one wave for each piece of the batch
    sources = [np.arange(1) for _ in range(batch_size)]

    # NOTE: generate infinit wave batches
    while True:
        batch = \
            np.zeros((batch_size, num_step_samples + 1, 512), dtype=np.float32)

        for index in range(batch_size):
            while sources[index].size < num_step_samples + 1:
                sources[index] = next(waves)

            for t in range(num_step_samples + 1):
                c_pre = sources[index][t] // 256
                f_pre = sources[index][t] % 256

                batch[index, t, sources[index][t] // 256] = 1.0
                batch[index, t, sources[index][t]  % 256] = 1.0

            sources[index] = sources[index][num_step_samples:]

        yield batch

