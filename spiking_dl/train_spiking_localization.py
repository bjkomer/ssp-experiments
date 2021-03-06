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
parser.add_argument('--n-epochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--plot-vis-set', action='store_true')
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine', 'ang-rmse'])
parser.add_argument('--n-sensors', type=int, default=36)
parser.add_argument('--fov', type=int, default=360)
parser.add_argument('--max-dist', type=float, default=10, help='maximum distance for distance sensor')

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
    'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_100000samples/coloured_sensor_dataset.npz'
)
data = np.load(dataset_file)
train_input, train_output, test_input, test_output = create_localization_train_test_sets(
    data=data,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    args=args,
    n_mazes_to_use=args.n_mazes,
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
    inp = nengo.Node(np.zeros((36*4 + args.maze_id_dim,)))

    x = nengo_dl.Layer(tf.keras.layers.Dense(units=args.hidden_size))(inp)
    x = nengo_dl.Layer(neuron_type)(x)

    out = nengo_dl.Layer(tf.keras.layers.Dense(units=args.dim))(x)

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

# small validation set, so it will not take long to compute
val_input = test_input[:args.n_validation_samples]
val_output = test_output[:args.n_validation_samples]
val_input = val_input[:, None, :]
val_output = val_output[:, None, :]

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

# sim.compile(loss={out_p_filt: mse_loss})
sim.compile(
    loss={out_p_filt: mse_loss},
    # metrics={out_p_filt: angular_rmse},
)
first_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)
print("Loss before training:", first_eval["loss"])
# print("Angular RMSE before training:", first_eval["out_p_filt_angular_rmse"])

param_file = "./saved_params/localization_params_{}_hs{}_{}samples_{}epochs".format(
    args.loss_function, args.hidden_size, args.n_train_samples, args.n_epochs
)
history_file = "./saved_params/localization_train_history_{}_hs{}_{}samples_{}epochs.npz".format(
    args.loss_function, args.hidden_size, args.n_train_samples, args.n_epochs
)

if not os.path.exists(param_file + '.npz'):
    print("Training")
    # run training
    if args.loss_function == 'mse':
        sim.compile(
            # optimizer=tf.optimizers.RMSprop(0.001),
            optimizer=tf.optimizers.Adam(0.001),
            # loss={out_p: tf.losses.MSE()}
            loss={out_p: mse_loss},
        )
    elif args.loss_function == 'cosine':
        sim.compile(
            # optimizer=tf.optimizers.RMSprop(0.001),
            optimizer=tf.optimizers.Adam(0.001),
            loss={out_p: cosine_loss},
        )
    elif args.loss_function == 'ang-rmse':
        sim.compile(
            # optimizer=tf.optimizers.RMSprop(0.001),
            optimizer=tf.optimizers.Adam(0.001),
            loss={out_p: angular_rmse},
        )
    history = sim.fit(
        train_input,
        {out_p: train_output},
        epochs=args.n_epochs,
        validation_data=(val_input, val_output)
    )
    np.savez(
        history_file,
        **history.history,
    )
    print("Saving parameters to: {}".format(param_file))
    # save the parameters to file
    sim.save_params(param_file)
else:
    # load parameters
    print("Loading pre-trained parameters")
    sim.load_params(param_file)

sim.compile(
    loss={out_p_filt: mse_loss},
    # metrics={out_p_filt: angular_rmse},
)

final_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)

print("Loss after training:", final_eval["loss"])

# print("Angular RMSE after training:", final_eval["out_p_filt_angular_rmse"])

if args.plot_vis_set:

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
        encoding_func=encoding_func,
    )

    n_batches = vis_input.shape[0]
    batch_size = vis_input.shape[1]

    # vis_input = np.tile(vis_input[:, None, :], (1, n_steps, 1))
    # vis_output = np.tile(vis_output[:, None, :], (1, n_steps, 1))

    print("Running visualization")
    # viz_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)

    fig, ax = plt.subplots(1, n_batches)

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

        # computing 'predicted' coordinates, where the agent thinks it is
        predictions[:, :] = ssp_to_loc_v(
            batch_data,
            hmv, xs, ys
        )

        plot_predictions_v(
            predictions=predictions, coords=coords,
            ax=ax[bi],
            min_val=limit_low,
            max_val=limit_high,
            fixed_axes=True,
        )

    plt.show()

