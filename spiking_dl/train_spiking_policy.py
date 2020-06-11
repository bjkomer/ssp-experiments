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

import nengo_dl

parser = argparse.ArgumentParser('Train a spiking policy network')

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--maze-id-dim', type=int, default=256)
parser.add_argument('--net-seed', type=int, default=13)
parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=50000, help='Number of testing samples')
parser.add_argument('--n-mazes', type=int, default=10)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--n-epochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--plot-vis-set', action='store_true')

parser = add_encoding_params(parser)

args = parser.parse_args()

args.dim = 256
args.spatial_encoding = 'sub-toroid-ssp'

if not os.path.exists('saved_params'):
    os.makedirs('saved_params')

limit_low = 0
limit_high = 13
encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

home = os.path.expanduser("~")
dataset_file = os.path.join(
    home,
    'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
)
data = np.load(dataset_file)
train_input, train_output, test_input, test_output = create_policy_train_test_sets(
    data=data,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    args=args,
    n_mazes=args.n_mazes,
    encoding_func=encoding_func,
)

print("\nData Generation Complete\n")

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
print(test_output.shape)
# add single timestep to training data
train_input = train_input[:, None, :]
# train_output = train_output[:, None, None]
train_output = train_output[:, None, :]

# number of timesteps to repeat the input/output for
n_steps = 30
test_input = np.tile(test_input[:, None, :], (1, n_steps, 1))
# test_output = np.tile(test_output[:, None, None], (1, n_steps, 1))
test_output = np.tile(test_output[:, None, :], (1, n_steps, 1))


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


print(test_output.shape)

sim.compile(loss={out_p_filt: mse_loss})
sim.compile(
    loss={out_p_filt: mse_loss},
    metrics={out_p_filt: angular_rmse},
)
first_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)
print("Loss before training:", first_eval["loss"])
print("Angular RMSE before training:", first_eval["out_p_filt_angular_rmse"])

param_file = "./saved_params/policy_params_{}samples_{}epochs".format(args.n_train_samples, args.n_epochs)

if not os.path.exists(param_file + '.npz'):
    print("Training")
    # run training
    sim.compile(
        # optimizer=tf.optimizers.RMSprop(0.001),
        optimizer=tf.optimizers.Adam(0.001),
        # loss={out_p: tf.losses.MSE()}
        loss={out_p: mse_loss},
    )
    sim.fit(train_input, {out_p: train_output}, epochs=args.n_epochs)
    print("Saving parameters to: {}".format(param_file))
    # save the parameters to file
    sim.save_params(param_file)
else:
    # load parameters
    print("Loading pre-trained parameters")
    sim.load_params(param_file)

sim.compile(
    loss={out_p_filt: mse_loss},
    metrics={out_p_filt: angular_rmse},
)

final_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)

print("Loss after training:", final_eval["loss"])

print("Angular RMSE after training:", final_eval["out_p_filt_angular_rmse"])

if args.plot_vis_set:

    print("Generating visualization set")

    vis_input, vis_output, batch_size = create_policy_vis_set(
        data=data,
        args=args,
        n_mazes=args.n_mazes,
        encoding_func=encoding_func,
        maze_indices=[0, 1, 2, 3],
        goal_indices=[0, 1],
    )

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
