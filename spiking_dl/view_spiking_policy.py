import argparse
import nengo
import tensorflow as tf
import numpy as np
import nengo.spa as spa
import matplotlib.pyplot as plt
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from utils import create_policy_train_test_sets, compute_angular_rmse, create_policy_vis_set
from ssp_navigation.utils.path import plot_path_predictions, plot_path_predictions_image, get_path_predictions_image
import os
import sys

import nengo_dl

parser = argparse.ArgumentParser('Train a spiking policy network')

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--maze-id-dim', type=int, default=256)
parser.add_argument('--net-seed', type=int, default=13)
# parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
# parser.add_argument('--n-test-samples', type=int, default=50000, help='Number of testing samples')
# parser.add_argument('--n-validation-samples', type=int, default=5000, help='Number of test samples for validation')
parser.add_argument('--n-mazes', type=int, default=10)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--n-epochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--plot-vis-set', action='store_true')
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine', 'ang-rmse'])
parser.add_argument('--param-file', type=str, default='')

parser = add_encoding_params(parser)

args = parser.parse_args()

if args.param_file == '':
    print("Must specify --param-file")
    sys.exit(0)

args.dim = 256
args.spatial_encoding = 'sub-toroid-ssp'


limit_low = 0
limit_high = 13
encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

home = os.path.expanduser("~")
dataset_file = os.path.join(
    home,
    'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
)
data = np.load(dataset_file)
# train_input, train_output, test_input, test_output = create_policy_train_test_sets(
#     data=data,
#     n_train_samples=args.n_train_samples,
#     n_test_samples=args.n_test_samples,
#     args=args,
#     n_mazes=args.n_mazes,
#     encoding_func=encoding_func,
# )

# print("\nData Generation Complete\n")

with nengo.Network(seed=args.net_seed) as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)


    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    nengo_dl.configure_settings(stateful=False)

    # the input node that will be used to feed in (context, location, goal)
    inp = nengo.Node(np.zeros((args.dim*2 + args.maze_id_dim,)))

    x = nengo_dl.Layer(tf.keras.layers.Dense(units=args.hidden_size))(inp)
    x = nengo_dl.Layer(neuron_type)(x)

    out = nengo_dl.Layer(tf.keras.layers.Dense(units=2))(x)

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# minibatch_size = 200
minibatch_size = 256
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

print("\nSimulator Built\n")
# print(test_output.shape)
# # add single timestep to training data
# train_input = train_input[:, None, :]
# # train_output = train_output[:, None, None]
# train_output = train_output[:, None, :]
#
# # small validation set, so it will not take long to compute
# val_input = test_input[:args.n_validation_samples]
# val_output = test_output[:args.n_validation_samples]
# val_input = val_input[:, None, :]
# val_output = val_output[:, None, :]
#
# # number of timesteps to repeat the input/output for
# n_steps = 30
# test_input = np.tile(test_input[:, None, :], (1, n_steps, 1))
# # test_output = np.tile(test_output[:, None, None], (1, n_steps, 1))
# test_output = np.tile(test_output[:, None, :], (1, n_steps, 1))


def mse_loss(y_true, y_pred):
    return tf.metrics.MSE(
        y_true[:, -1], y_pred[:, -1]
    )


def cosine_loss(y_true, y_pred):
    return tf.metrics.CosineSimilarity(
        y_true[:, -1], y_pred[:, -1]
    )


def angular_rmse(y_true, y_pred):
    return compute_angular_rmse(
        y_true[:, -1], y_pred[:, -1]
    )


# param_file = "./saved_params/policy_params_{}_hs{}_{}samples_{}epochs".format(
#     args.loss_function, args.hidden_size, args.n_train_samples, args.n_epochs
# )
# history_file = "./saved_params/train_history_{}_hs{}_{}samples_{}epochs.npz".format(
#     args.loss_function, args.hidden_size, args.n_train_samples, args.n_epochs
# )

param_file = args.param_file

# load parameters
print("Loading pre-trained parameters")
sim.load_params(param_file)

print(type(sim))

# sim.compile(
#     loss={out_p_filt: mse_loss},
#     metrics={out_p_filt: angular_rmse},
# )
#
# final_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)
#
# print("Loss after training:", final_eval["loss"])
#
# print("Angular RMSE after training:", final_eval["out_p_filt_angular_rmse"])


print("Generating visualization set")

vis_input, vis_output, batch_size = create_policy_vis_set(
    data=data,
    args=args,
    n_mazes=args.n_mazes,
    encoding_func=encoding_func,
    maze_indices=[0, 1, 2, 3],
    goal_indices=[0, 1],
    # TEMP: for debugging
    # maze_indices=[0, ],
    # goal_indices=[2, 3, 4, 5, 6, 7, 8],
    # x_offset=0.25,
    # y_offset=0.25,
)

n_steps = 30
vis_input = np.tile(vis_input[:, None, :], (1, n_steps, 1))
# vis_output = np.tile(vis_output[:, None, :], (1, n_steps, 1))

print("Running visualization")
# viz_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)

n_batches = 4*2

for bi in range(n_batches):
    viz_eval = sim.predict(vis_input[bi*batch_size:(bi+1)*batch_size])
    # batch_data = viz_eval[out_p_filt][:, -1, :]
    batch_data = viz_eval[out_p_filt][:, 10:, :].mean(axis=1)
    directions = vis_output[bi*batch_size:(bi+1)*batch_size, :]

    print('pred.shape', batch_data.shape)
    print('directions.shape', directions.shape)

    wall_overlay = (directions[:, 0] == 0) & (directions[:, 1] == 0)

    print('wall_overlay.shape', wall_overlay.shape)

    fig_truth, rmse = plot_path_predictions_image(
        directions_pred=directions,
        directions_true=directions,
        wall_overlay=wall_overlay
    )

    fig_pred, rmse = plot_path_predictions_image(
        directions_pred=batch_data,
        directions_true=directions,
        wall_overlay=wall_overlay
    )

    plt.show()

    print(batch_data)
    print(batch_data.shape)
