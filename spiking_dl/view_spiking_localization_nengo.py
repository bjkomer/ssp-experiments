import argparse
import nengo
import tensorflow as tf
import numpy as np
import nengo.spa as spa
import matplotlib.pyplot as plt
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params, get_encoding_heatmap_vectors
from utils import create_localization_train_test_sets, compute_angular_rmse, create_localization_viz_set
from ssp_navigation.utils.path import plot_path_predictions, plot_path_predictions_image, get_path_predictions_image
from spatial_semantic_pointers.utils import ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions_v
import os
import sys
import pickle

import nengo_dl

parser = argparse.ArgumentParser('Train a spiking localization network')

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--maze-id-dim', type=int, default=256)
parser.add_argument('--net-seed', type=int, default=13)
parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=50000, help='Number of testing samples')
parser.add_argument('--n-validation-samples', type=int, default=5000, help='Number of test samples for validation')
parser.add_argument('--n-mazes', type=int, default=10)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--n-layers', type=int, default=1, choices=[1, 2])
parser.add_argument('--n-epochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--plot-vis-set', action='store_true')
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine', 'ang-rmse'])
parser.add_argument('--n-sensors', type=int, default=36)
parser.add_argument('--fov', type=int, default=360)
parser.add_argument('--max-dist', type=float, default=10, help='maximum distance for distance sensor')

parser.add_argument('--param-file', type=str, default='')

parser = add_encoding_params(parser)

args = parser.parse_args()


if args.param_file == '':
    print("Must specify --param-file")
    sys.exit(0)

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
    'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_100000samples/coloured_sensor_dataset.npz'
)
data = np.load(dataset_file)


with nengo.Network(seed=args.net_seed) as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)


    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    net.config[nengo.Connection].transform = nengo_dl.dists.Glorot()
    nengo_dl.configure_settings(stateful=False)

    # the input node that will be used to feed in (context, location, goal)
    inp = nengo.Node(np.zeros((36*4 + args.maze_id_dim,)))

    if '.pkl' in args.param_file:
        print("Loading values from pickle file")

        localization_params = pickle.load(open(args.param_file, 'rb'))
        localization_inp_params = localization_params[0]
        localization_ens_params = localization_params[1]
        localization_out_params = localization_params[2]

        hidden_ens = nengo.Ensemble(
            n_neurons=args.hidden_size,
            # dimensions=36*4 + args.maze_id_dim,
            dimensions=1,
            neuron_type=neuron_type,
            **localization_ens_params
        )

        out = nengo.Node(size_in=args.dim)

        if args.n_layers == 1:

            conn_in = nengo.Connection(
                inp, hidden_ens.neurons, synapse=None,
                **localization_inp_params
            )
            conn_out = nengo.Connection(
                hidden_ens.neurons, out, synapse=None,
                # function=lambda x: np.zeros((args.dim,)),
                **localization_out_params
            )
        elif args.n_layers == 2:

            localization_mid_params = localization_params[3]
            localization_ens_two_params = localization_params[4]

            hidden_ens_two = nengo.Ensemble(
                n_neurons=args.hidden_size,
                dimensions=1,
                neuron_type=neuron_type,
                **localization_ens_two_params
            )

            conn_in = nengo.Connection(
                inp, hidden_ens.neurons, synapse=None,
                **localization_inp_params
            )
            conn_mid = nengo.Connection(
                hidden_ens.neurons, hidden_ens_two.neurons, synapse=None,
                **localization_mid_params
            )
            conn_out = nengo.Connection(
                hidden_ens_two.neurons, out, synapse=None,
                **localization_out_params
            )

    else:
        hidden_ens = nengo.Ensemble(
            n_neurons=args.hidden_size,
            dimensions=36 * 4 + args.maze_id_dim,
            neuron_type=neuron_type
        )

        out = nengo.Node(size_in=args.dim)

        if args.n_layers == 2:
            hidden_ens_two = nengo.Ensemble(
                n_neurons=args.hidden_size,
                dimensions=1,
                # dimensions=36*4 + args.maze_id_dim,
                neuron_type=neuron_type
            )

            conn_in = nengo.Connection(
                inp, hidden_ens.neurons, synapse=None
            )

            conn_mid = nengo.Connection(
                hidden_ens.neurons, hidden_ens_two.neurons, synapse=None
            )

            conn_out = nengo.Connection(
                hidden_ens_two.neurons, out, synapse=None,
            )

        else:

            conn_in = nengo.Connection(
                inp, hidden_ens.neurons, synapse=None
            )
            conn_out = nengo.Connection(
                hidden_ens.neurons, out, synapse=None,
            )

        # conn_in = nengo.Connection(inp, hidden_ens, synapse=None)
        # conn_out = nengo.Connection(hidden_ens, out, synapse=None, function=lambda x: np.zeros((args.dim,)))

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# minibatch_size = 200
minibatch_size = 256
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

print("\nSimulator Built\n")

# number of timesteps to repeat the input/output for
n_steps = 30




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



# load parameters
if '.pkl' not in args.param_file:
    # load parameters
    print("Loading pre-trained npz parameters")
    sim.load_params(args.param_file)

sim.compile(
    loss={out_p_filt: mse_loss},
    # metrics={out_p_filt: angular_rmse},
)

res = 64
xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

coords = np.zeros((res * res, 2))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        coords[i * res + j, 0] = x
        coords[i * res + j, 1] = y

print("Generating visualization set")

vis_input, vis_output = create_localization_viz_set(
    args=args,
    n_mazes_to_use=2,  # 10
    encoding_func=encoding_func,
)

n_batches = vis_input.shape[0]
batch_size = vis_input.shape[1]

# vis_input = np.tile(vis_input[:, None, :], (1, n_steps, 1))
# vis_output = np.tile(vis_output[:, None, :], (1, n_steps, 1))

print("Running visualization")
# viz_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)

fig, ax = plt.subplots(2, n_batches)

for bi in range(n_batches):
    vis_batch_input = np.tile(vis_input[bi, :, None, :], (1, n_steps, 1))
    viz_eval = sim.predict(vis_batch_input)
    # batch_data = viz_eval[out_p_filt][:, -1, :]
    batch_data = viz_eval[out_p_filt][:, 10:, :].mean(axis=1)
    true_ssps = vis_output[bi, :, :]

    print('pred.shape', batch_data.shape)
    print('true_ssps.shape', true_ssps.shape)

    wall_overlay = np.sum(true_ssps, axis=1) == 0

    print('wall_overlay.shape', wall_overlay.shape)

    hmv = get_encoding_heatmap_vectors(xs, ys, args.dim, encoding_func, normalize=False)

    predictions = np.zeros((res * res, 2))
    truth = np.zeros((res * res, 2))

    # computing 'predicted' coordinates, where the agent thinks it is
    predictions[:, :] = ssp_to_loc_v(
        batch_data,
        hmv, xs, ys
    )

    truth[:, :] = ssp_to_loc_v(
        true_ssps,
        hmv, xs, ys
    )

    plot_predictions_v(
        predictions=predictions[wall_overlay == False, :], coords=coords[wall_overlay == False, :],
        ax=ax[0, bi],
        min_val=limit_low,
        max_val=limit_high,
        fixed_axes=True,
    )

    plot_predictions_v(
        predictions=truth[wall_overlay == False, :], coords=coords[wall_overlay == False, :],
        ax=ax[1, bi],
        min_val=limit_low,
        max_val=limit_high,
        fixed_axes=True,
    )

plt.show()
