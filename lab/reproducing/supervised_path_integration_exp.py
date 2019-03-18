import argparse
import numpy as np
from arguments import add_parameters
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

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

n_samples = 1000

rollout_length = 100

velocity_inputs = np.zeros((n_samples, rollout_length, 3))
activation_outputs = np.zeros((n_samples, n_place_cells + n_hd_cells))

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

# velocity_input = Input(shape=(3,))
# lstm_layer = LSTM(128)(velocity_input)
# linear_layer = Dense(512, activation='linear')(lstm_layer)
# pc_output = Dense(args.n_place_cells, activation='softmax')(linear_layer)
# hd_output = Dense(args.n_hd_cells, activation='softmax')(linear_layer)
#
# model = Model(inputs=velocity_input, outputs=[pc_output, hd_output])


model = Sequential()
# model.add(Input(shape=(3,)))
model.add(LSTM(128))
model.add(Dense(512))
model.add(Dense(args.n_place_cells + args.n_hd_cells))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
# model.compile(
#     optimizer='rmsprop',
#     loss={'pc_output': 'binary_crossentropy', 'hd_output': 'binary_crossentropy'},
#     metrics=['accuracy']
# )

# model.fit(tmp_input, tmp_output, batch_size=10)
model.fit(velocity_inputs, activation_outputs, batch_size=10)

print(model.evaluate(velocity_inputs, activation_outputs, verbose=0))

print(model.summary())

# model.fit(x_train, y_train, batch_size=16, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=16)