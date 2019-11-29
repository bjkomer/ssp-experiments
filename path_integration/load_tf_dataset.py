import tensorflow as tf
import numpy as np
from tfrecord import (
    Writer, Reader,
    pack_int64_list, unpack_int64_list,
    pack_float_list, unpack_float_list,
    pack_bytes_list, unpack_bytes_list,
)

fname = '/home/bjkomer/deepmind/data/grid-cells-datasets/square_room_100steps_2.2m_1000000'

fname_individual = '/home/bjkomer/deepmind/data/grid-cells-datasets/square_room_100steps_2.2m_1000000/0000-of-0099.tfrecord'

FILENAME = fname_individual

# example['init_pos'], example['init_hd'],
# example['ego_vel'][:, :self._steps, :],
# example['target_pos'][:, :self._steps, :],
# example['target_hd'][:, :self._steps, :]

sequence_length = 100

feature_map = {
    'init_pos':
        tf.FixedLenFeature(shape=[2], dtype=tf.float32),
    'init_hd':
        tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    'ego_vel':
        tf.FixedLenFeature(
            shape=[sequence_length, 3],
            dtype=tf.float32),
    'target_pos':
        tf.FixedLenFeature(
            shape=[sequence_length, 2],
            dtype=tf.float32),
    'target_hd':
        tf.FixedLenFeature(
            shape=[sequence_length, 1],
            dtype=tf.float32),
}

feature_shapes = {
    'init_pos':
        (2,),
    'init_hd':
        (1,),
    'ego_vel':
        (sequence_length, 3),
    'target_pos':
        (sequence_length, 2),
    'target_hd':
        (sequence_length, 1),
}


def reshape_sample(sample):
    reshaped = {}
    for key, value in sample.items():
        reshaped[key] = np.array(value).reshape(feature_shapes[key])
    return reshaped

def unpack_sample(feats):
    return {
        # 'floats': unpack_float_list(feats['floats']),
        # 'bytes' : unpack_bytes_list(feats['bytes'])
        'init_pos': unpack_float_list(feats['init_pos']),
        'init_hd': unpack_float_list(feats['init_hd']),
        'ego_vel': unpack_float_list(feats['ego_vel']),
        'target_pos': unpack_float_list(feats['target_pos']),
        'target_hd': unpack_float_list(feats['target_hd']),
    }


with Reader(FILENAME, unpack_sample) as r:
    for sample in r:
        print(reshape_sample(sample))
        # print(sample)
        # # print(len(sample['init_pos']))
        # # print(len(sample['init_hd']))
        # print(len(sample['ego_vel']))
        # print(len(sample['target_pos']))
        # print(len(sample['target_hd']))
        # print("")

        # example = tf.parse_example(sample, feature_map)
        # print(example)
