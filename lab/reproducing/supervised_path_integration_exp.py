import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
from arguments import add_parameters
import torch
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, TimeDistributed
from keras.layers import Embedding
from keras.layers import LSTM
import keras.backend as K


parser = argparse.ArgumentParser('Run 2D supervised path integration experiment')

parser = add_parameters(parser)

args = parser.parse_args()

data = np.load('data/path_integration_trajectories_200t_15s.npz')

positions = data['positions']
angles = data['angles']
lin_vels = data['lin_vels']
ang_vels = data['ang_vels']
cos_ang_vels = np.cos(ang_vels)
sin_ang_vels = np.sin(ang_vels)
pc_activations = data['pc_activations']
hd_activations = data['hd_activations']

n_trajectories = positions.shape[0]
trajectory_length = positions.shape[1]
n_place_cells = pc_activations.shape[2]
n_hd_cells = hd_activations.shape[2]

print("trajectory_lengths", trajectory_length)

tmp_input = np.zeros((10, 100, 3))
# tmp_output = np.zeros((10, 100, n_place_cells + n_hd_cells))
tmp_output = np.zeros((10, n_place_cells + n_hd_cells))

batch_size = 10

n_samples = 1000

rollout_length = 100

velocity_inputs = np.zeros((n_samples, rollout_length, 3))
activation_outputs = np.zeros((n_samples, n_place_cells + n_hd_cells))
pc_outputs = np.zeros((n_samples, n_place_cells))
hd_outputs = np.zeros((n_samples, n_hd_cells))

# these include outputs for every time-step
full_pc_outputs = np.zeros((n_samples, rollout_length, n_place_cells))
full_hd_outputs = np.zeros((n_samples, rollout_length, n_hd_cells))

initial_states = np.zeros((n_samples, 3))
pc_inputs = np.zeros((n_samples, n_place_cells))
hd_inputs = np.zeros((n_samples, n_hd_cells))


for i in range(n_samples):
    # choose random trajectory
    traj_ind = np.random.randint(low=0, high=n_trajectories)
    # choose random segment of trajectory
    step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length)

    # index of final step of the trajectory
    step_ind_final = step_ind + 99

    velocity_inputs[i, :, 0] = lin_vels[traj_ind, step_ind:step_ind_final+1]
    velocity_inputs[i, :, 1] = cos_ang_vels[traj_ind, step_ind:step_ind_final + 1]
    velocity_inputs[i, :, 2] = sin_ang_vels[traj_ind, step_ind:step_ind_final + 1]

    activation_outputs[i, :n_place_cells] = pc_activations[traj_ind, step_ind_final]
    activation_outputs[i, n_place_cells:] = hd_activations[traj_ind, step_ind_final]
    pc_outputs[i, :] = pc_activations[traj_ind, step_ind_final]
    hd_outputs[i, :] = hd_activations[traj_ind, step_ind_final]

    full_pc_outputs[i, :] = pc_activations[traj_ind, step_ind:step_ind_final + 1]
    full_hd_outputs[i, :] = hd_activations[traj_ind, step_ind:step_ind_final + 1]

    # initial state of the LSTM is a linear transform of the ground truth place and hd cell activations
    pc_inputs[i, :] = pc_activations[traj_ind, step_ind]
    hd_inputs[i, :] = hd_activations[traj_ind, step_ind]

    # # initial state of the LSTM. Set to ground truth starting position
    # initial_states[i, 0] = positions[traj_ind, step_ind, 0]
    # initial_states[i, 1] = positions[traj_ind, step_ind, 1]
    # initial_states[i, 3] = angles[traj_ind, step_ind]

print("checking max values")
print(np.max(velocity_inputs))
print(np.max(activation_outputs))
print("")

# velocity_input = Input(shape=(3,))
# velocity_input = Input(name='velocity_input',  batch_shape=(batch_size, rollout_length, 3,))
# lstm_layer = LSTM(128, stateful=True, batch_input_shape=(batch_size, rollout_length, 3))(velocity_input)
# lstm_layer = LSTM(128, stateful=False, initial_state=initial_state_input)(velocity_input)

pc_input = Input(name='hd_input', shape=(n_place_cells,))
hd_input = Input(name='hd_input', shape=(n_hd_cells,))

velocity_input = Input(name='velocity_input',  shape=(rollout_length, 3,))
# lstm_layer = LSTM(128, stateful=False, initial_state=initial_state_input)(velocity_input)
lstm_layer = LSTM(128, stateful=False, return_sequences=True)(velocity_input)
linear_layer = TimeDistributed(Dense(512, activation='linear'))(lstm_layer)
pc_output = TimeDistributed(Dense(args.n_place_cells, activation='softmax'), name='pc_output')(linear_layer)
hd_output = TimeDistributed(Dense(args.n_hd_cells, activation='softmax'), name='hd_output')(linear_layer)

model = Model(inputs=velocity_input, outputs=[pc_output, hd_output])

model.compile(
    optimizer='rmsprop',
    loss={'pc_output': 'binary_crossentropy', 'hd_output': 'binary_crossentropy'},
    metrics=['accuracy']
)

model.fit(
    # {'velocity_input': velocity_inputs, 'initial_state_input'},
    {'velocity_input': velocity_inputs,},
    # {'pc_output': pc_outputs, 'hd_output': hd_outputs},
    {'pc_output': full_pc_outputs, 'hd_output': full_hd_outputs},
    epochs=3,
    batch_size=batch_size,
)

# print(model.evaluate(velocity_inputs, activation_outputs, verbose=0))
print(
    model.evaluate(
        {'velocity_input': velocity_inputs},
        # {'pc_output': pc_outputs, 'hd_output': hd_outputs},
        {'pc_output': full_pc_outputs, 'hd_output': full_hd_outputs},
        verbose=0
    )
)

#
# model = Sequential()
# # model.add(Input(shape=(3,)))
# model.add(LSTM(128, activation='linear'))
# model.add(Dense(512, activation='linear'))
# model.add(Dense(args.n_place_cells + args.n_hd_cells, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# # model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

print(model)


# model = Graph()
# model.add_input(name='velocity_input', input_shape=(3,))
#
# model.add_node(name='lstm_layer', layer=LSTM(128), input='velocity_input')
#
# model.add_node(name='linear_layer', layer=Dense(512, activation='linear'), input='lstm_layer')
#
# model.add_node(name='pc_output_layer', layer=Dense(args.n_place_cells, activation='softmax'), input='linear_layer')
# model.add_node(name='hd_output_layer', layer=Dense(args.n_hd_cells, activation='softmax'), input='linear_layer')
#
# model.add_output(name='pc_output', input='pc_output_layer')
# model.add_output(name='hd_output', input='hd_output_layer')
#


# model.fit(tmp_input, tmp_output, batch_size=10)
# model.fit(velocity_inputs, activation_outputs, batch_size=10)
#
# print(model.evaluate(velocity_inputs, activation_outputs, verbose=0))

print(model.summary())

# model.fit(x_train, y_train, batch_size=16, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=16)