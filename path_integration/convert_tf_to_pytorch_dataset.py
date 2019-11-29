import tensorflow as tf
import numpy as np
from tfrecord import (
    Writer, Reader,
    pack_int64_list, unpack_int64_list,
    pack_float_list, unpack_float_list,
    pack_bytes_list, unpack_bytes_list,
)

fname = 'data/path_integration_tf_trajectories.npz'
sequence_length = 100
n_records = 100
record_size = 10000
n_samples = 1000000

assert(n_records * record_size == n_samples)

# positions = np.zeros((args.n_trajectories, trajectory_steps, 2))
# angles = np.zeros((args.n_trajectories, trajectory_steps))
# lin_vels = np.zeros((args.n_trajectories, trajectory_steps))
# ang_vels = np.zeros((args.n_trajectories, trajectory_steps))

# This database will have a different format from the others, so will need its own dataloader later
init_pos = np.zeros((n_samples, 2))
init_hd = np.zeros((n_samples, 1))
ego_vel = np.zeros((n_samples, sequence_length, 3))
target_pos = np.zeros((n_samples, sequence_length, 2))
target_hd = np.zeros((n_samples, sequence_length, 1))

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
        'init_pos': unpack_float_list(feats['init_pos']),
        'init_hd': unpack_float_list(feats['init_hd']),
        'ego_vel': unpack_float_list(feats['ego_vel']),
        'target_pos': unpack_float_list(feats['target_pos']),
        'target_hd': unpack_float_list(feats['target_hd']),
    }


tf_record_base = '/home/bjkomer/deepmind/data/grid-cells-datasets/square_room_100steps_2.2m_1000000/00{:02d}-of-0099.tfrecord'

for ri in range(n_records):
    print("record {} of {}".format(ri+1, n_records))
    tf_record_name = tf_record_base.format(ri)
    with Reader(tf_record_name, unpack_sample) as r:
        si = 0
        for sample in r:
            data = reshape_sample(sample)

            init_pos[ri * record_size + si, :] = data['init_pos']
            init_hd[ri * record_size + si, :] = data['init_hd']
            ego_vel[ri * record_size + si, :, :] = data['ego_vel']
            target_pos[ri * record_size + si, :, :] = data['target_pos']
            target_hd[ri * record_size + si, :, :] = data['target_hd']

            si += 1

np.savez(
    fname,
    init_pos=init_pos,
    init_hd=init_hd,
    ego_vel=ego_vel,
    target_pos=target_pos,
    target_hd=target_hd,
)

# np.savez(
#     fname,
#     positions=positions,
#     angles=angles,
#     lin_vels=lin_vels,
#     ang_vels=ang_vels,
#     cartesian_vels=cartesian_vels,
#     env_size=args.env_size,
# )

print("data saved to: {}".format(fname))
