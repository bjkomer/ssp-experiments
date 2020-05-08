# this is a simplified version with no learning phase, the layout is given each trial as an SSP
import nengo
import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, encode_point_hex, get_heatmap_vectors_hex
from world import ItemMap, ExperimentControl
from collections import OrderedDict
import argparse
import os
import nengo_spa as spa

parser = argparse.ArgumentParser('Run a mental map task with Nengo')

parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--neurons-per-dim', type=int, default=5)
parser.add_argument('--n-cconv-neurons', type=int, default=50)
parser.add_argument('--n-items', type=int, default=7)
parser.add_argument('--duration', type=int, default=50)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--env-size', type=int, default=10)
parser.add_argument('--time-per-item', type=int, default=3)

args = parser.parse_args()

if not os.path.exists('output'):
    os.makedirs('output')

fname = 'output/kosslyn_ssp_simplified_{}dim_{}seed_{}items_{}size_{}duration'.format(
    args.dim, args.seed, args.n_items, args.env_size, args.duration
)

items = OrderedDict({"Tree": np.array([1, 2]),
                     "Pond": np.array([4, 4]),
                     "Well": np.array([6, 1]),
                     "Rock": np.array([2, 7]),
                     "Reed": np.array([9, 3]),
                     "Lake": np.array([8, 8]),
                     "Bush": np.array([1, 3]),
                     })

randomize = False
# item_vocab = spa.Vocabulary(args.dim, randomize=randomize)
item_vocab = spa.Vocabulary(args.dim)

item_vocab.populate(';'.join(list(items.keys())))

rng = np.random.RandomState(seed=args.seed)
X = make_good_unitary(args.dim, rng=rng)
Y = make_good_unitary(args.dim, rng=rng)
Z = make_good_unitary(args.dim, rng=rng)

mem_sp = spa.SemanticPointer(data=np.zeros((args.dim,)))
for key, value in items.items():
    mem_sp += item_vocab[key] * encode_point_hex(value[0], value[1], X, Y, Z)


def encode_func(pos):
    return encode_point_hex(pos[0], pos[1], X, Y, Z).v


xs = np.linspace(0, args.env_size, 256)
hmv = get_heatmap_vectors_hex(xs, xs, X, Y, Z)


def decode_func(ssp):
    vs = np.tensordot(ssp, hmv, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = xs[xy[1]]

    return np.array([x, y])


model = nengo.Network()
with model:
    # currently visualized location
    current_loc = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # direction to move
    direction = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # memory of the map layout
    memory = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # mental visual field
    vision = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # the item to be scanned to
    queried_item = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    mem_input = nengo.Node(
        lambda t: mem_sp.v
    )
    nengo.Connection(mem_input, memory, synapse=None)

    # nengo.Connection(memory, cconv)
    # nengo.Connection(queried_item, cconv)
    # nengo.Connection(cconv, desired_loc)

    # move the representation based on the direction
    cconv_move = nengo.networks.CircularConvolution(n_neurons=args.neurons_per_dim, dimensions=args.dim)
    nengo.Connection(current_loc, cconv_move.input_a)
    nengo.Connection(direction, cconv_move.input_b)
    nengo.Connection(cconv_move.output, current_loc)

    # experiment node
    exp_node = nengo.Node(
        ExperimentControl(
            items=items,
            vocab=item_vocab,
            file_name=fname,
            time_per_item=args.time_per_item,
            num_test_pairs=50,
            ssp_dir=True,
            encode_func=encode_func,
            decode_func=decode_func,
        ),
        size_in=args.dim + args.dim,
        size_out=args.dim + args.dim,
    )

    nengo.Connection(exp_node[:args.dim], direction)
    nengo.Connection(exp_node[args.dim:], queried_item)
    nengo.Connection(vision, exp_node[args.dim:])
    nengo.Connection(current_loc, exp_node[:args.dim])

    # p_exp = nengo.Probe(exp_node)


sim = nengo.Simulator(model)
sim.run(args.duration)

# np.savez(
#     fname,
#     data=sim.data[p_exp]
# )
