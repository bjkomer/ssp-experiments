# this is a simplified version with no learning phase, the layout is given each trial as an SSP
import nengo
import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, encode_point, encode_point_hex, get_heatmap_vectors, get_heatmap_vectors_hex
from spatial_semantic_pointers.plots import SpatialHeatmap
# from ssp_navigation.utils.encoding import add_encoding_params
from world import ItemMap, ExperimentControl
from collections import OrderedDict
import argparse
import os
import nengo_spa as spa
from spatial_semantic_pointers.networks.ssp_cleanup import SpatialCleanup
# softlinked from neural_implementation
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir, SSPState
import pickle

parser = argparse.ArgumentParser('Run a mental map task with Nengo')

# parser = add_encoding_params(parser)
parser.add_argument('--dim', type=int, default=512) #272
parser.add_argument('--neurons-per-dim', type=int, default=50)
parser.add_argument('--n-cconv-neurons', type=int, default=50)
parser.add_argument('--n-items', type=int, default=7)
parser.add_argument('--duration', type=int, default=50)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--env-size', type=int, default=10)
parser.add_argument('--time-per-item', type=float, default=1.0) #3
parser.add_argument('--sim-thresh', type=float, default=0.4)
parser.add_argument('--dir-mag-limit', type=float, default=1.0)

if not os.path.exists('output'):
    os.makedirs('output')

if __name__ == '__main__':
    args = parser.parse_args()
    # fname = 'output/kosslyn_ssp_simplified_{}dim_{}seed_{}items_{}size_{}duration'.format(
    #     args.dim, args.seed, args.n_items, args.env_size, args.duration
    # )
    fname = 'output/attractor_kosslyn_ssp_cconv_{}seed_tpi{}_thresh{}_vel{}_npd{}'.format(
        args.seed, args.time_per_item, args.sim_thresh, args.dir_mag_limit, args.neurons_per_dim
    )
else:
    args = parser.parse_args([])
    fname = 'output/debug'



axis_rng = np.random.RandomState(seed=14)

T = (args.dim+1)//2
n_toroid = T // 3

phis = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)
angles = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)

X, Y = orthogonal_hex_dir(phis=phis, angles=angles, even_dim=True)

args.dim = len(X.v)

print(args.dim)

if args.seed == -1:
    # fixed demo locations
    items = OrderedDict({"Tree": np.array([1, 2]),
                         "Pond": np.array([4, 4]),
                         "Well": np.array([6, 1]),
                         "Rock": np.array([2, 7]),
                         "Reed": np.array([9, 3]),
                         "Lake": np.array([8, 8]),
                         "Bush": np.array([1, 3]),
                         })
else:
    # random locations
    rng_items = np.random.RandomState(seed=args.seed)
    # don't pick items right on the edge of the bounds
    locs = rng_items.uniform(-4.5, 4.5, size=(7, 2))
    items = OrderedDict({"Tree": locs[0, :],
                         "Pond": locs[1, :],
                         "Well": locs[2, :],
                         "Rock": locs[3, :],
                         "Reed": locs[4, :],
                         "Lake": locs[5, :],
                         "Bush": locs[6, :]
                         })


randomize = False
# item_vocab = spa.Vocabulary(args.dim, randomize=randomize)
item_vocab = spa.Vocabulary(args.dim)

item_vocab.populate(';'.join(list(items.keys())))

limit = 5
res = 256
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

mem_sp = spa.SemanticPointer(data=np.zeros((args.dim,)))
for key, value in items.items():
    # mem_sp += item_vocab[key] * encode_point_hex(value[0], value[1], X, Y, Z)
    mem_sp += item_vocab[key] * encode_point(value[0], value[1], X, Y)


def encode_func(pos):
    # return encode_point_hex(pos[0], pos[1], X, Y, Z).v
    return encode_point(pos[0], pos[1], X, Y).v


# xs = np.linspace(-1, args.env_size+1, 256)
if not os.path.exists('hmv_attractor_exp_{}.npz'.format(args.dim)):
    # hmv = get_heatmap_vectors_hex(xs, xs, X, Y, Z)
    hmv = get_heatmap_vectors(xs, xs, X, Y)
    np.savez('hmv_exp_{}.npz'.format(args.dim), hmv=hmv)
else:
    hmv = np.load('hmv_attractor_exp_{}.npz'.format(args.dim))['hmv']


def decode_func(ssp):
    vs = np.tensordot(ssp, hmv, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = xs[xy[1]]

    return np.array([x, y])


model = nengo.Network(seed=13)
spiking_dir = True
model.config[nengo.Ensemble].neuron_type = nengo.LIF()
with model:
    # currently visualized location
    model.current_loc = SSPState(
        vocab=args.dim,
        phis=phis,
        angles=angles,
        feedback=1.01,
        limit_low=-limit,
        limit_high=limit,
    )

    # direction to move
    model.direction = spa.State(args.dim)

    # memory of the map layout
    model.memory = spa.State(args.dim)

    # mental visual field (just using the output of the cconv directly)
    # vision = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # the item to be scanned to
    model.queried_item = spa.State(args.dim)

    mem_input = nengo.Node(
        lambda t: mem_sp.v
    )
    nengo.Connection(mem_input, model.memory.input, synapse=None)

    # computing what the agent is currently imagining
    cconv_view = nengo.networks.CircularConvolution(n_neurons=args.neurons_per_dim*2, dimensions=args.dim, invert_b=True)
    nengo.Connection(model.memory.output, cconv_view.input_a)
    nengo.Connection(model.current_loc.output, cconv_view.input_b)

    # move the representation based on the direction
    cconv_move = nengo.networks.CircularConvolution(n_neurons=args.neurons_per_dim*2, dimensions=args.dim)
    nengo.Connection(model.current_loc.output, cconv_move.input_a)
    nengo.Connection(model.direction.output, cconv_move.input_b)
    nengo.Connection(cconv_move.output, model.current_loc.input)

    # # Cleanup a potentially noisy spatial semantic pointer to a clean version
    # spatial_cleanup = nengo.Node(
    #     SpatialCleanup(model_path=ssp_cleanup_path, dim=args.dim, hidden_size=512),
    #     size_in=args.dim, size_out=args.dim
    # )
    # nengo.Connection(cconv_move.output, spatial_cleanup)
    # nengo.Connection(spatial_cleanup, model.current_loc.input)

    # experiment node
    exp_node = nengo.Node(
        ExperimentControl(
            items=items,
            vocab=item_vocab,
            file_name=fname,
            time_per_item=args.time_per_item,
            num_test_pairs=50,
            sim_thresh=args.sim_thresh,
            dir_mag_limit=args.dir_mag_limit,
            ssp_dir=True,
            encode_func=encode_func,
            decode_func=decode_func,
        ),
        size_in=2 * args.dim,
        size_out=3 * args.dim,
    )

    if spiking_dir:
        # direction_params = pickle.load(open('dir_params.pkl', 'rb'))
        direction_params = pickle.load(open('dir_rel_params.pkl', 'rb'))
        direction_inp_params = direction_params[0]
        direction_ens_params = direction_params[1]
        direction_out_params = direction_params[2]
        dir_comp_ens = nengo.Ensemble(
            n_neurons=8192,
            dimensions=1,
            neuron_type=nengo.LIF(amplitude=0.01),
            **direction_ens_params
        )
        cur_dir = nengo.Node(size_in=args.dim*2, size_out=args.dim*2)
        nengo.Connection(cconv_view.output, cur_dir[:args.dim])
        nengo.Connection(model.current_loc.output, cur_dir[args.dim:])
        nengo.Connection(dir_comp_ens.neurons, model.direction.input, **direction_out_params)
        # nengo.Connection(cconv_view.output, dir_comp_ens.neurons, **direction_inp_params)
        nengo.Connection(cur_dir, dir_comp_ens.neurons, **direction_inp_params)
    else:
        nengo.Connection(exp_node[:args.dim], model.direction.input)
        nengo.Connection(cconv_view.output, exp_node[args.dim:])

    nengo.Connection(exp_node[args.dim:2*args.dim], model.queried_item.input)
    # drive the memory to the next start when the current task is finished
    nengo.Connection(exp_node[2 * args.dim:], model.current_loc.input)

    nengo.Connection(model.current_loc.output, exp_node[:args.dim])

    initial_kick = nengo.Node(lambda t: 1 if t < 0.01 else 0)
    nengo.Connection(initial_kick, model.current_loc.input[0])

    if __name__ != '__main__':
        # plots for debugging
        heatmap = SpatialHeatmap(heatmap_vectors=hmv, xs=xs, ys=xs, cmap='plasma', vmin=-1, vmax=1)
        heatmap_node = nengo.Node(
            heatmap,
            size_in=args.dim,
            size_out=0
        )
        nengo.Connection(model.current_loc.output, heatmap_node)

        dir_heatmap = SpatialHeatmap(heatmap_vectors=hmv, xs=xs, ys=xs, cmap='plasma', vmin=-1, vmax=1)
        dir_heatmap_node = nengo.Node(
            heatmap,
            size_in=args.dim,
            size_out=0
        )
        nengo.Connection(model.direction.output, dir_heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model)
    sim.run(args.duration)

