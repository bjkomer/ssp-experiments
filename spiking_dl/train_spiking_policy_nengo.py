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
import pickle

import nengo_dl

parser = argparse.ArgumentParser('Train a spiking policy network')

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--maze-id-dim', type=int, default=256)
parser.add_argument('--net-seed', type=int, default=13)
parser.add_argument('--n-train-samples', type=int, default=5000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=5000, help='Number of testing samples')
parser.add_argument('--n-validation-samples', type=int, default=5000, help='Number of test samples for validation')
parser.add_argument('--n-mazes', type=int, default=10)
parser.add_argument('--hidden-size', type=int, default=1024)
parser.add_argument('--n-layers', type=int, default=1, choices=[1, 2])
parser.add_argument('--n-epochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--plot-vis-set', action='store_true')
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine', 'ang-rmse'])

parser.add_argument('--input-noise', type=float, default=0.01, help='Gaussian noise to add to the inputs when training')
parser.add_argument('--shift-noise', type=float, default=0.2,
                    help='Shifting coordinates before encoding by up to this amount as a form of noise')

parser.add_argument('--weight-reg', type=float, default=0.001)

parser.add_argument('--neuron-type', type=str, default='lif', choices=['lif', 'relu', 'lifrate'])

parser = add_encoding_params(parser)

args = parser.parse_args()

# args.dim = 256
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
    input_noise=args.input_noise,
    shift_noise=args.shift_noise,
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
    net.config[nengo.Connection].transform = nengo_dl.dists.Glorot()

    if args.neuron_type == 'lif':
        neuron_type = nengo.LIF(amplitude=0.01)
    elif args.neuron_type == 'relu':
        neuron_type = nengo.RectifiedLinear()
    elif args.neuron_type == 'lifrate':
        neuron_type = nengo.LIFRate(amplitude=0.01)
    else:
        raise NotImplementedError

    nengo_dl.configure_settings(stateful=False)
    # nengo_dl.configure_settings(keep_history=True)
    nengo_dl.configure_settings(keep_history=False)

    # the input node that will be used to feed in (context, location, goal)
    inp = nengo.Node(np.zeros((args.dim*2 + args.maze_id_dim,)))

    hidden_ens = nengo.Ensemble(
        n_neurons=args.hidden_size,
        # dimensions=args.dim*2 + args.maze_id_dim,
        dimensions=1,
        neuron_type=neuron_type
    )

    out = nengo.Node(size_in=2)

    out_p = nengo.Probe(out, label="out_p")
    net.config[out_p].keep_history = False

    if args.n_layers == 1:

        conn_in = nengo.Connection(inp, hidden_ens.neurons, synapse=None)
        conn_out = nengo.Connection(hidden_ens.neurons, out, synapse=None)
    elif args.n_layers == 2:

        hidden_ens_two = nengo.Ensemble(
            n_neurons=args.hidden_size,
            dimensions=1,
            neuron_type=neuron_type
        )

        conn_in = nengo.Connection(inp, hidden_ens.neurons, synapse=None)
        conn_mid = nengo.Connection(hidden_ens.neurons, hidden_ens_two.neurons, synapse=None)
        conn_out = nengo.Connection(hidden_ens_two.neurons, out, synapse=None)

        p_weight_mid = nengo.Probe(conn_mid, "weights", label="p_weight_mid")
        net.config[p_weight_mid].keep_history = False
    else:
        raise NotImplementedError

    p_weight_in = nengo.Probe(conn_in, "weights", label="p_weight_in")
    net.config[p_weight_in].keep_history = False
    p_weight_out = nengo.Probe(conn_out, "weights", label="p_weight_out")
    net.config[p_weight_out].keep_history = False



    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# minibatch_size = 200
minibatch_size = 256

with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:

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
    # sim.compile(
    #     loss={out_p_filt: mse_loss},
    #     metrics={out_p_filt: angular_rmse},
    # )
    # first_eval = sim.evaluate(test_input, {out_p_filt: test_output}, verbose=0)
    # print("Loss before training:", first_eval["loss"])
    # print("Angular RMSE before training:", first_eval["out_p_filt_angular_rmse"])


    suffix = '{}dim_{}layer_{}_hs{}_{}samples_{}epochs_{}reg'.format(
        args.dim, args.n_layers, args.loss_function, args.hidden_size, args.n_train_samples, args.n_epochs, args.weight_reg
    )

    if args.neuron_type == 'relu':
        suffix += '_relu'
    if args.neuron_type == 'lifrate':
        suffix += 'lifrate'

    param_file = "./saved_params/nengo_policy_params_{}".format(
        suffix
    )
    history_file = "./saved_params/nengo_policy_train_history_{}.npz".format(
        suffix
    )
    nengo_obj_file = "./saved_params/nengo_policy_obj_{}.pkl".format(
        suffix
    )

    if not os.path.exists(param_file + '.npz'):
        print("Training")
        # run training
        if args.loss_function == 'mse':
            if args.weight_reg > 0:
                if args.n_layers == 1:
                    sim.compile(
                        # optimizer=tf.optimizers.RMSprop(0.001),
                        optimizer=tf.optimizers.Adam(0.001),
                        # loss={out_p: tf.losses.MSE()}
                        loss={
                            out_p: mse_loss,
                            p_weight_in: nengo_dl.losses.Regularize(),
                            p_weight_out: nengo_dl.losses.Regularize(),
                        },
                        loss_weights={
                            out_p: 1,
                            p_weight_in: args.weight_reg,
                            p_weight_out: args.weight_reg,
                        }
                    )
                elif args.n_layers == 2:
                    sim.compile(
                        # optimizer=tf.optimizers.RMSprop(0.001),
                        optimizer=tf.optimizers.Adam(0.001),
                        # loss={out_p: tf.losses.MSE()}
                        loss={
                            out_p: mse_loss,
                            p_weight_in: nengo_dl.losses.Regularize(),
                            p_weight_mid: nengo_dl.losses.Regularize(),
                            p_weight_out: nengo_dl.losses.Regularize(),
                        },
                        loss_weights={
                            out_p: 1,
                            p_weight_in: args.weight_reg,
                            p_weight_mid: args.weight_reg,
                            p_weight_out: args.weight_reg,
                        }
                    )
                else:
                    raise NotImplementedError
            else:
                print("compiling with no weight reg")
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
        else:
            raise NotImplementedError

        if args.weight_reg > 0:
            dummy_train_array = np.empty((train_input.shape[0], 1, 0), dtype=bool)
            dummy_val_array = np.empty((val_input.shape[0], 1, 0), dtype=bool)
            if args.n_layers == 1:
                history = sim.fit(
                    x=train_input,
                    y={
                        out_p: train_output,
                        p_weight_in: dummy_train_array,
                        p_weight_out: dummy_train_array,
                        # p_weight_in:  np.ones((train_input.shape[0], 1, p_weight_in.size_in)),
                        # p_weight_out: np.ones((train_input.shape[0], 1, p_weight_in.size_in)),
                    },
                    epochs=args.n_epochs,
                    validation_data=(
                        val_input,
                        {
                            out_p: val_output,
                            p_weight_in: dummy_val_array,
                            p_weight_out: dummy_val_array,
                            # p_weight_in: np.ones((val_input.shape[0], 1, p_weight_in.size_in)),
                            # p_weight_out: np.ones((val_input.shape[0], 1, p_weight_in.size_in)),
                        }
                    )
                )
            else:
                history = sim.fit(
                    x=train_input,
                    y={
                        out_p: train_output,
                        p_weight_in: dummy_train_array,
                        p_weight_mid: dummy_train_array,
                        p_weight_out: dummy_train_array,
                        # p_weight_in: np.ones((train_input.shape[0], 1, p_weight_in.size_in)),
                        # p_weight_mid: np.ones((train_input.shape[0], 1, p_weight_in.size_in)),
                        # p_weight_out: np.ones((train_input.shape[0], 1, p_weight_in.size_in)),
                    },
                    epochs=args.n_epochs,
                    validation_data=(
                        val_input,
                        {
                            out_p: val_output,
                            p_weight_in: dummy_val_array,
                            p_weight_mid: dummy_val_array,
                            p_weight_out: dummy_val_array,
                            # p_weight_in: np.ones((val_input.shape[0], 1, p_weight_in.size_in)),
                            # p_weight_mid: np.ones((val_input.shape[0], 1, p_weight_in.size_in)),
                            # p_weight_out: np.ones((val_input.shape[0], 1, p_weight_in.size_in)),
                        }
                    )
                )
        else:
            print("about to fit")
            # history = sim.fit(
            #     x=train_input,
            #     y={
            #         out_p: train_output,
            #     },
            #     epochs=args.n_epochs,
            #     validation_data=(val_input, val_output)
            # )
            history = sim.fit(
                train_input,
                {
                    out_p: train_output,
                },
                epochs=args.n_epochs,
                validation_data=(
                    val_input,
                    {
                        out_p: val_output,
                    }
                )
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
        metrics={out_p_filt: angular_rmse},
    )


    if args.n_layers == 1:
        params = sim.get_nengo_params([conn_in, hidden_ens, conn_out])
    elif args.n_layers == 2:
        params = sim.get_nengo_params([conn_in, hidden_ens, conn_out, conn_mid, hidden_ens_two])
    else:
        raise NotImplementedError

    pickle.dump(params, open(nengo_obj_file, "wb"))
    print("Nengo Parameters Saved")

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
