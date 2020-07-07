import nengo
import argparse
import pickle
import numpy as np
import nengo.spa as spa
import nengo_spa
import os
import torch
from utils import NengoGridEnv, generate_cleanup_dataset, PolicyGT, LocalizationGT
from ssp_navigation.utils.encodings import get_encoding_function, get_encoding_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
# from spatial_semantic_pointers.utils import encode_point, get_heatmap_vectors


parser = argparse.ArgumentParser('Spiking Integrated System')

parser.add_argument('--policy-hidden-size', type=int, default=2048)
parser.add_argument('--policy-n-layers', type=int, default=2)
parser.add_argument('--localization-hidden-size', type=int, default=4096)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--maze-id-dim', type=int, default=256)
parser.add_argument('--spatial-encoding', type=str, default='sub-toroid-ssp')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--scale-ratio', type=float, default=0.0)
parser.add_argument('--n-proj', type=int, default=3)
# parser.add_argument('--ssp-scaling', type=float, default=0.5)
parser.add_argument('--ssp-scaling', type=float, default=1.0)

parser.add_argument('--maze-index', type=int, default=1)
parser.add_argument('--env-seed', type=int, default=13)
# parser.add_argument('--noise', type=float, default=0.0) #.25
parser.add_argument('--noise', type=float, default=0.25)
parser.add_argument('--normalize-action', action='store_true')
parser.add_argument('--use-dataset-goals', action='store_true')
parser.add_argument('--use-localization-gt', action='store_true')
parser.add_argument('--use-cleanup-gt', action='store_true')
parser.add_argument('--use-policy-gt', action='store_true')
parser.add_argument('--use-precomp-policy', action='store_true')
parser.add_argument('--use-precomp-localization', action='store_true')
# parser.add_argument('--n-goals', type=int, default=7)
parser.add_argument('--n-goals', type=int, default=2)

# For the cleanup
# parser.add_argument('--neurons-per-dim', type=int, default=50)
parser.add_argument('--neurons-per-dim', type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.normalize_action = True
    # args.use_localization_gt = True
    args.use_cleanup_gt = True
    # args.use_policy_gt = True
    # args.use_dataset_goals = True
    # args.use_precomp_policy = True
    # args.use_precomp_localization = True

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
maze_sps = np.zeros((10, args.maze_id_dim))
# overwrite data
for mi in range(10):
    maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v


# Load learned parameters for components of the network
# TODO: update these to final trained versions
# policy_param_file = 'saved_params/nengo_obj_mse_hs1024_1000000samples_250epochs.pkl'
# policy_param_file = 'saved_params/nengo_policy_obj_mse_hs1024_1500000samples_50epochs.pkl'
# policy_param_file = 'saved_params/nengo_policy_obj_mse_hs1024_2000000samples_100epochs.pkl'
# policy_param_file = 'saved_params/nengo_policy_obj_mse_hs2048_5000000samples_100epochs.pkl'
# policy_param_file = 'saved_params/nengo_policy_obj_2layer_mse_hs2048_2500000samples_100epochs_1e-05reg.pkl'
policy_param_file = 'saved_params/nengo_policy_obj_2layer_mse_hs2048_5000000samples_100epochs_1e-05reg.pkl'

# localization_param_file = 'saved_params/nengo_localization_obj_mse_hs1024_1000000samples_250epochs.pkl'
# localization_param_file = 'saved_params/nengo_localization_obj_mse_hs1024_5000000samples_100epochs.pkl'
# localization_param_file = 'saved_params/nengo_localization_obj_1layer_mse_hs4096_2500000samples_50epochs_1e-06reg.pkl'
localization_param_file = 'saved_params/nengo_localization_obj_1layer_mse_hs4096_3000000samples_25epochs_1e-05reg.pkl'

policy_params = pickle.load(open(policy_param_file, 'rb'))
policy_inp_params = policy_params[0]
policy_ens_params = policy_params[1]
policy_out_params = policy_params[2]
localization_params = pickle.load(open(localization_param_file, 'rb'))
localization_inp_params = localization_params[0]
localization_ens_params = localization_params[1]
localization_out_params = localization_params[2]


limit_low = 0
limit_high = 13
res = 64
# xs and ys for html plots only
xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)
encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

cache_fname = 'saved_params/neural_cleanup_dataset_{}.npz'.format(args.dim)
if os.path.exists(cache_fname):
    data = np.load(cache_fname)
    clean_vectors = data['clean_vectors']
    noisy_vectors = data['noisy_vectors']
    coords = data['coords']
    heatmap_vectors = data['heatmap_vectors']
else:
    # heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
    heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, args.dim, encoding_func, normalize=False)

    n_samples = 250
    n_items = 7
    clean_vectors, noisy_vectors, coords = generate_cleanup_dataset(
        encoding_func,
        n_samples,
        args.dim,
        n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(limit_low, limit_high, limit_low, limit_high),
        seed=13,
        normalize_memory=True
    )

    np.savez(
        cache_fname,
        clean_vectors=clean_vectors,
        noisy_vectors=noisy_vectors,
        coords=coords,
        heatmap_vectors=heatmap_vectors,
    )

x_axis_vec = encoding_func(1, 0)
y_axis_vec = encoding_func(0, 1)
x_axis_sp = nengo_spa.SemanticPointer(data=x_axis_vec)
y_axis_sp = nengo_spa.SemanticPointer(data=y_axis_vec)


def to_ssp(v):
    return encoding_func(x=v[0], y=v[1])
    # return encoding_func(x=v[0]+10.0, y=v[1]+10.0)


def to_ssp_loc(v):
    return encoding_func(x=v[0] + 1, y=v[1] + 1)


def normalize(v):
    length = np.linalg.norm(v)
    if length > 0:
        return v / length
    else:
        return v


model = nengo.Network(seed=13)
neuron_type = nengo.LIF(amplitude=0.01)
synapse = 0.002
# neuron_type = nengo.LIF()
# model.config[nengo.Connection].synapse = None

with model:

    # Gridworld environment wrapped in a nengo node
    # input is the 2D movement direction
    # output is the 256D memory, 256D context vector and 36*4 sensor measurements
    nengo_grid_env = NengoGridEnv(
        maze_sps=maze_sps,
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp,
        maze_index=args.maze_index,
        env_seed=args.env_seed,
        noise=args.noise,
        n_goals=args.n_goals,
        normalize_action=args.normalize_action,
        use_dataset_goals=args.use_dataset_goals,
        nengo_dt=0.001,
        sim_dt=0.001,#0.01,  # how often to call the gridworld simulator
    )
    env = nengo.Node(
        nengo_grid_env,
        size_in=2,
        size_out=args.maze_id_dim + args.dim + 36*4 + 4  # the last 4 dimensions are for debugging
    )

    ###########
    # Cleanup #
    ###########
    spatial_cleanup = nengo.Ensemble(
        n_neurons=args.dim * args.neurons_per_dim, dimensions=args.dim, neuron_type=nengo.LIF()
    )

    #TODO: cconv currently happening in env atm
    nengo.Connection(env[args.dim:args.dim*2], spatial_cleanup)

    ################
    # Localization #
    ################

    localization_inp_node = nengo.Node(
        size_in=args.maze_id_dim + 36*4,
        size_out=args.maze_id_dim + 36*4
    )

    # localization_out_node = nengo.Node(
    #     size_in=args.dim,
    #     size_out=args.dim
    # )
    localization_out_node = nengo.Ensemble(
        n_neurons=1,
        dimensions=args.dim,
        neuron_type=nengo.Direct()
    )

    localization_ens = nengo.Ensemble(
        n_neurons=args.localization_hidden_size,
        # dimensions=args.maze_id_dim + 36*4,
        dimensions=1,
        neuron_type=neuron_type,
        **localization_ens_params
    )

    # Context
    nengo.Connection(
        env[:args.dim],
        localization_inp_node[36*4:]
        # localization_ens[0:args.dim],
        # synapse=None,
        # **localization_inp_params
    )

    # Sensors
    nengo.Connection(
        env[2*args.dim:2*args.dim+36*4],
        # localization_inp_node[args.dim:],
        localization_inp_node[:36*4],
        # localization_ens[args.dim:],
        # synapse=None,
        # **localization_inp_params
    )

    nengo.Connection(
        localization_inp_node,
        localization_ens.neurons,
        **localization_inp_params
    )

    ##########
    # Policy #
    ##########

    policy_inp_node = nengo.Node(
        size_in=args.dim*2 + args.maze_id_dim,
        size_out=args.dim*2 + args.maze_id_dim
    )

    policy_out_node = nengo.Node(
        size_in=2,
        size_out=2,
    )
    if args.use_policy_gt or args.use_precomp_policy:
        if args.use_precomp_policy:
            # pre-computed spiking policy
            path = 'ssp_navigation_sandbox/spiking_dl/spiking_policy_solution.npz'
        else:
            # ground truth dataset
            path = 'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
        policy_ens = nengo.Node(
            PolicyGT(
                heatmap_vectors=heatmap_vectors, maze_index=args.maze_index, dim=args.dim,
                path=path,
            ),
            size_in=args.dim*2 + args.maze_id_dim,
            size_out=2
        )
        nengo.Connection(
            policy_inp_node,
            policy_ens,
            synapse=synapse
        )
        nengo.Connection(
            policy_ens,
            policy_out_node,
            synapse=0.001,
        )
    else:
        if args.policy_n_layers == 1:
            policy_ens = nengo.Ensemble(
                n_neurons=args.policy_hidden_size,
                # dimensions=args.dim*2 + args.maze_id_dim,
                dimensions=1,
                neuron_type=neuron_type,
                **policy_ens_params
            )

            nengo.Connection(
                policy_inp_node,
                policy_ens.neurons,
                synapse=None,
                **policy_inp_params
            )

            nengo.Connection(
                policy_ens.neurons,
                policy_out_node,
                # synapse=0.1,
                synapse=0.001,
                **policy_out_params
            )
        elif args.policy_n_layers == 2:
            policy_mid_params = policy_params[3]
            policy_ens_two_params = policy_params[4]

            policy_ens = nengo.Ensemble(
                n_neurons=args.policy_hidden_size,
                dimensions=1,
                neuron_type=neuron_type,
                **policy_ens_params
            )

            policy_ens_two = nengo.Ensemble(
                n_neurons=args.policy_hidden_size,
                dimensions=1,
                neuron_type=neuron_type,
                **policy_ens_two_params
            )

            conn_in = nengo.Connection(
                policy_inp_node, policy_ens.neurons,
                synapse=synapse,
                **policy_inp_params
            )
            conn_mid = nengo.Connection(
                policy_ens.neurons, policy_ens_two.neurons,
                synapse=synapse,
                **policy_mid_params
            )
            conn_out = nengo.Connection(
                policy_ens_two.neurons, policy_out_node,
                synapse=synapse,
                **policy_out_params
            )

    # Context
    nengo.Connection(
        env[:args.dim],
        policy_inp_node[0:args.dim],
        # synapse=None,
        # **policy_inp_params
    )

    # Location
    if args.use_precomp_localization:
        loc_ens = nengo.Node(
            LocalizationGT(
                heatmap_vectors=heatmap_vectors, maze_index=args.maze_index, dim=args.dim,
                cleanup=True,
                # cleanup=False,
                path='ssp_navigation_sandbox/spiking_dl/spiking_localization_solution.npz',
            ),
            size_in=36*4 + args.maze_id_dim,
            size_out=args.dim
        )
        nengo.Connection(
            localization_inp_node,
            loc_ens,
            synapse=synapse
        )
        nengo.Connection(
            loc_ens,
            localization_out_node,
            synapse=0.01,
            # synapse=0.1,
        )
    elif args.use_localization_gt:
        nengo.Connection(
            env[[-4, -3]],
            # policy_inp_node[args.dim:2*args.dim],
            localization_out_node,
            # function=to_ssp_loc,
            function=to_ssp,
            # synapse=0.1,
            # synapse=None,
        )
    else:
        # connection from localization network
        # TODO: should there be a cleanup added on here? hopefully learned model is clean enough
        nengo.Connection(
            localization_ens.neurons,
            localization_out_node,
            # policy_inp_node[args.dim:2*args.dim],
            # function=lambda x: np.zeros((args.dim,)),
            # synapse=0.1,
            synapse=0.01,
            **localization_out_params
        )
    nengo.Connection(
        localization_out_node,
        policy_inp_node[args.dim:2 * args.dim],
        function=normalize,
        synapse=None
    )

    # Goal
    if args.use_cleanup_gt:
        nengo.Connection(
            env[[-2, -1]],
            policy_inp_node[2 * args.dim:3 * args.dim],
            function=to_ssp,
            # synapse=None,
        )
    else:
    # cleanup connection
        nengo.Connection(
            spatial_cleanup,
            policy_inp_node[2*args.dim:3*args.dim],
            function=clean_vectors,
            eval_points=noisy_vectors,
            scale_eval_points=False,
            # solver=LstsqL2(),
        )

    # Velocity output
    # nengo.Connection(
    #     policy_ens.neurons,
    #     env,
    #     # function=lambda x: [0, 0],
    #     synapse=0.1,
    #     **policy_out_params
    # )


    nengo.Connection(
        policy_out_node,
        env,
        synapse=None,
    )


    # SSP visualization
    if __name__ != "__main__":
        vmin=0
        vmax=1
        spatial_cleanup_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=args.dim, size_out=0,
        )
        nengo.Connection(spatial_cleanup, spatial_cleanup_heatmap_node)

        localization_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=args.dim, size_out=0,
        )
        nengo.Connection(
            localization_out_node,
            localization_heatmap_node,
            function=normalize,
            synapse=None,
        )

        goal_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=args.dim, size_out=0,
        )
        nengo.Connection(policy_inp_node[2*args.dim:3*args.dim], goal_heatmap_node)
