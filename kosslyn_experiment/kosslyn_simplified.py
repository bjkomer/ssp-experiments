# this is a simplified version with no learning phase, the layout is given each trial as an SSP
import nengo
import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, encode_point_hex
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

fname = 'output/kosslyn_simplified_{}dim_{}seed_{}items_{}size_{}duration'.format(
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


model = nengo.Network()
with model:
    # currently visualized location
    current_loc = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # memory of the map layout
    memory = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # mental visual field
    vision = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # the item to be scanned to
    queried_item = nengo.Ensemble(n_neurons=args.dim*args.neurons_per_dim, dimensions=args.dim)

    # mental world
    item_map = ItemMap(
        shape=(args.env_size, args.env_size),
        items=items,
        vocab=item_vocab,
        item_rad=.1,
        motion_type='holonomic'  # 'teleport'
    )
    env = nengo.Node(item_map, size_in=2, size_out=2 + args.dim)

    # experiment node
    exp_node = nengo.Node(
        ExperimentControl(
            items=items,
            vocab=item_vocab,
            file_name=fname,
            time_per_item=args.time_per_item,
            num_test_pairs=50,
            ssp_dir=False
        ),
        size_in=2 + args.dim,
        size_out=2 + args.dim,
    )

    nengo.Connection(exp_node[[0, 1]], env[[0, 1]])
    nengo.Connection(env, exp_node)

    # p_exp = nengo.Probe(exp_node)


sim = nengo.Simulator(model)
sim.run(args.duration)

# np.savez(
#     fname,
#     data=sim.data[p_exp]
# )
