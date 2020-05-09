# This version quickly adds in SSPs instead of the 'surface'
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, ssp_to_loc, get_heatmap_vectors
from utils import orthogonal_hex_dir_7dim
import os

# Associate place and item using the Voja learning rule
# Learn the associations both ways simultaneously. Can query either location or item
# this version has a bunch of stuff cut out/direct mode to run faster

# TODO: at 'test time' query an item, which will give a location,
#       slowly move the location query from the current location to that items location
#       this can be done by taking the vector difference as the direction, and using a
#       unit velocity vector for movement in that direction. Could also use the distance
#       directly and have it saturate
#       this simulates the 'scanning' across the map
#       stop once the recalled item is close enough to the queried item
#       log the time that it takes, and repeat the experiment for a bunch of pairs

# TODO: make more populations use neurons

import sys

import nengo
import nengo.spa as spa
import numpy as np
# NOTE: this version uses the older ones from utils instead of world
from utils import ItemMap, ExperimentControl
from collections import OrderedDict
from utils import EncoderPlot, WeightPlot
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run a model for the Kosslyn mental map task")

    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--num-test-pairs', type=int, default=50)
    parser.add_argument('--time-per-item', type=float, default=2.)
    parser.add_argument('--dimensionality', type=int, default=7)
    parser.add_argument('--phi', type=float, default=0.2)
    parser.add_argument('--randomize_vectors', action='store_true')
    parser.add_argument('--file-name', type=str, default="output/exp_data_kosslyn_low_dim_hex")

    args = parser.parse_args()

    seed = args.seed
    num_test_pairs = args.num_test_pairs
    time_per_item = args.time_per_item
    randomize = args.randomize_vectors
    D = args.dimensionality
    file_name = args.file_name
    phi = np.pi * args.phi
else:
    seed = 13
    num_test_pairs = 50
    time_per_item = 2
    randomize = False
    D = 7  # 16
    file_name = "output/exp_data_kosslyn_low_dim_hex"
    phi = np.pi * 0.2

dim = 7


random_inputs = False

# if values are normalized before going into the voja rule
normalize = True

items = OrderedDict({"Tree": np.array([1, 2]),
                     "Pond": np.array([4, 4]),
                     "Well": np.array([6, 1]),
                     "Rock": np.array([2, 7]),
                     "Reed": np.array([9, 3]),
                     "Lake": np.array([8, 8]),
                     "Bush": np.array([1, 3]),
                     })

shape = (10, 10)

model = spa.SPA(seed=seed)

action_vocab = spa.Vocabulary(3, randomize=False)

item_vocab = spa.Vocabulary(D, randomize=randomize)
# item_vocab = spa.Vocabulary(len(items), randomize=True)

"""
keys = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1],
         ])
"""

keys = np.eye(len(items))

for i, f in enumerate(items.keys()):
    item_vocab.add(f, keys[i])  # fixed vector
    # item_vocab.parse(f) # random vector
# for f in items.keys():
#    item_vocab.add(f,len(items))

# Explore the environment and learn from it. 'Command' controls the desired position
action_vocab.add('EXPLORE', [1, 0, 0])

# Recall locations or items based on queries
action_vocab.add('RECALL', [0, 1, 0])

# Move to the location of the queried item
action_vocab.add('FIND', [0, 0, 1])


rstate = np.random.RandomState(seed=seed)

# x_axis_sp = make_good_unitary(dim=dim, rng=rstate)
# y_axis_sp = make_good_unitary(dim=dim, rng=rstate)

x_axis_sp, y_axis_sp, sv = orthogonal_hex_dir_7dim(phi=phi)

# TODO: use full limit eventually
xs = np.linspace(0, 10, 256)
ys = np.linspace(0, 10, 256)

# TODO: cache the heatmap vectors
hmv_cache = 'hmv_cache_dim{}_phi{}_limit{}.npz'.format(dim, np.round(phi/np.pi, 2), 10)
if os.path.isfile(hmv_cache):
    print("Loading Heatmap Vectors from Cache")
    heatmap_vectors = np.load(hmv_cache)['heatmap_vectors']
else:
    print("Generating Heatmap Vectors")
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
    print("Saving Heatmap Vectors to Cache")
    np.savez(
        hmv_cache,
        heatmap_vectors=heatmap_vectors
    )


# compute a control signal to get to the location
def compute_velocity(x):
    # which way the agent should face to go directly to the target
    desired_ang = np.arctan2(-x[1], -x[0])

    ang_diff = -1 * (x[2] - desired_ang)

    if ang_diff > np.pi:
        ang_diff -= 2 * np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2 * np.pi

    ang_vel = ang_diff * 2.5
    if np.sqrt(x[0] ** 2 + x[1] ** 2) < .001:
        lin_vel = 0
        ang_vel = 0
    elif abs(ang_diff) < np.pi / 4.:
        lin_vel = 1.6 * np.sqrt(x[0] ** 2 + x[1] ** 2)
    elif abs(ang_diff) < np.pi / 2.:
        lin_vel = .8 * np.sqrt(x[0] ** 2 + x[1] ** 2)
    else:
        lin_vel = 0

    return lin_vel, ang_vel


# compute a fixed length vector in the direction of the input
def compute_direction_delta(x, delta_length=10):
    norm = np.linalg.norm(x)
    if norm == 0:
        # If at the target location, don't move
        direction = np.zeros(2)
    elif norm < .1:
        # If close to the target location, slow down
        direction = -x
    else:
        # If far from the target location, move with a constant velocity
        direction = -delta_length * x / norm

    return direction


# Scales the location output to be between -1 and 1
def scale_location(x):
    x_out = x[0] / (shape[0] / 2.) - 1
    y_out = x[1] / (shape[1] / 2.) - 1
    th_out = x[2] / np.pi  # TODO: should this get scaled at all?

    return [x_out, y_out, th_out]


def scale_xy_node(t, x):
    x_out = x[0] / (shape[0] / 2.) - 1
    y_out = x[1] / (shape[1] / 2.) - 1

    return x_out, y_out


def env_scale_node(t, x):
    x_out = (x[0] + 1) * (shape[0] / 2.)
    y_out = (x[1] + 1) * (shape[1] / 2.)

    return x_out, y_out


def loc_to_surface(x):
    x_in = x[0]
    y_in = x[1]

    return encode_point(x_in, y_in, x_axis_sp, y_axis_sp).v


def loc_scale_to_surface(x):
    # # scale to between -1 and 1
    # x_in = x[0] / (shape[0] / 2.) - 1
    # y_in = x[1] / (shape[1] / 2.) - 1

    # TEMP DEBUGGING
    x_in = x[0]
    y_in = x[1]

    return encode_point(x_in, y_in, x_axis_sp, y_axis_sp).v

    # # take sin and cos between -pi and pi
    # xx = np.cos(x_in * np.pi)
    # xy = np.sin(x_in * np.pi)
    #
    # yx = np.cos(y_in * np.pi)
    # yy = np.sin(y_in * np.pi)
    #
    # # NOTE: normalizing here changes the intercept from 1 to 0, huge improvement!
    # denom = np.sqrt(xx ** 2 + xy ** 2 + yx ** 2 + yy ** 2)
    # xx /= denom
    # xy /= denom
    # yx /= denom
    # yy /= denom
    #
    # return xx, xy, yx, yy


def surface_to_env(x):
    xy = ssp_to_loc(sp=x, heatmap_vectors=heatmap_vectors, xs=xs, ys=ys)
    # print(xy)
    return xy / (10 / 2) - 1

    # xp = np.arctan2(x[1], x[0]) / np.pi
    # yp = np.arctan2(x[3], x[2]) / np.pi
    #
    # return xp, yp


def normalize(x):
    if np.linalg.norm(x) > 0:
        return x / np.linalg.norm(x)
    else:
        return x


# temporary artificial basal ganglia
# x[0]: control signal
# x[[1,2]] : environment position
# x[[3,4]] : query position
def control_loc(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[3], x[4]
    else:
        return x[1], x[2]


def control_item(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[5], x[6], x[7], x[8]
    else:
        return x[1], x[2], x[3], x[4]


def voja_inhib_func(t, x):
    if x < -1:
        return -1
    else:
        return x


# intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
intercept = 0  # calculated from another script
intercept = .70  # calculated from another script
intercept_item_to_loc = .7  # calculated from another script
intercept_loc_to_item = .5  # 5.55111512313e-17 # calculated from another script

with model:
    # NOTE: FlavourLand is currently hardcoded for orthogonal non-random vectors
    # TODO: make a version that can be given a vocabulary
    item_map = ItemMap(
        shape=shape,
        items=items,
        vocab=item_vocab,
        item_rad=.1,
        motion_type='holonomic'  # 'teleport'
    )
    env = nengo.Node(item_map, size_in=3, size_out=3 + D)

    ec = ExperimentControl(
        items=items,
        vocab=item_vocab,
        time_per_item=time_per_item,
        num_test_pairs=num_test_pairs,  # 6+5+4+3+2+1 = 21 total pairs
        file_name=file_name,
    )

    exp_cont = nengo.Node(
        ec,
        size_in=D,
        size_out=2 + 3 + D
    )

    taste = nengo.Ensemble(n_neurons=10 * D, dimensions=D,
                           radius=1.2, neuron_type=nengo.LIF())

    """
    # linear and angular velocity
    if random_inputs:
        velocity = nengo.Node(RandomRun())
    else:
        velocity = nengo.Node([0,0])
    nengo.Connection(velocity[:2], env[:2])
    """
    nengo.Connection(env[3:], taste)

    voja_loc = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_item = nengo.Voja(post_tau=None, learning_rate=5e-2)

    memory_loc = nengo.Ensemble(n_neurons=400, dimensions=dim, intercepts=[intercept_loc_to_item] * 400)
    memory_item = nengo.Ensemble(
        n_neurons=D * 50,
        dimensions=D,
        intercepts=[intercept_item_to_loc] * (D * 50)
    )

    # Query a location to get a response for what item was there
    query_location = nengo.Node([0, 0])

    # Location currently being visualized on the map
    # Represented in encoded coordinates
    mental_location = nengo.Ensemble(n_neurons=400, dimensions=dim)

    # Decoded location, for display purposes
    mental_location_decoded = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

    nengo.Connection(mental_location, mental_location_decoded, function=surface_to_env)

    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type = nengo.Direct()
    with cfg:
        # Pick the item to query using a semantic pointer input
        model.query_item = spa.State(D, vocab=item_vocab)

        # Choose between learning, recall, and item finding
        model.action = spa.State(3, vocab=action_vocab)

    task = nengo.Node(size_in=1, size_out=1)

    # The position that the agent will try to move to using its controller
    desired_pos = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # TODO: switch from direct mode once things work
    working_loc = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())
    working_item = nengo.Ensemble(n_neurons=200, dimensions=D, neuron_type=nengo.Direct())

    if normalize:
        # NOTE: loc_to_surface already normalizes
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface,
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_item = nengo.Connection(working_item, memory_item, function=normalize,
                                        learning_rule_type=voja_item, synapse=None)
    else:
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface,
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_item = nengo.Connection(working_item, memory_item,
                                        learning_rule_type=voja_item, synapse=None)

    # makes sure the voja learning connection only receives 0 (learning) or -1 (not learning) exactly
    voja_inhib = nengo.Node(voja_inhib_func, size_in=1, size_out=1)

    nengo.Connection(task, voja_inhib, synapse=None, transform=-1)

    nengo.Connection(voja_inhib, conn_in_loc.learning_rule, synapse=None)
    nengo.Connection(voja_inhib, conn_in_item.learning_rule, synapse=None)

    # Try only learning when items present
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning, synapse=None)

    nengo.Connection(learning, voja_inhib, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1 * np.ones((1, D)))

    recall_item = nengo.Ensemble(n_neurons=200, dimensions=D, neuron_type=nengo.Direct())
    recall_loc = nengo.Ensemble(n_neurons=400, dimensions=dim, neuron_type=nengo.Direct())

    conn_out_item = nengo.Connection(memory_loc, recall_item,
                                     learning_rule_type=nengo.PES(1e-3),
                                     function=lambda x: np.random.random(D)
                                     )
    conn_out_loc = nengo.Connection(memory_item, recall_loc,
                                    learning_rule_type=nengo.PES(1e-3),
                                    function=lambda x: np.random.random(dim)
                                    )

    error_item = nengo.Ensemble(n_neurons=200, dimensions=len(items))
    error_loc = nengo.Ensemble(n_neurons=400, dimensions=dim)

    # Connect up error populations
    nengo.Connection(env[3:], error_item, transform=-1, synapse=None)
    nengo.Connection(recall_item, error_item, transform=1, synapse=None)
    nengo.Connection(error_item, conn_out_item.learning_rule)

    surface_loc = nengo.Node(size_in=dim, size_out=dim)
    nengo.Connection(env[:2], surface_loc, function=loc_scale_to_surface, synapse=None)
    nengo.Connection(surface_loc, error_loc, transform=-1, synapse=None)
    nengo.Connection(recall_loc, error_loc, transform=1, synapse=None)
    nengo.Connection(error_loc, conn_out_loc.learning_rule)

    # inhibit learning based on learning signal
    nengo.Connection(learning, error_item.neurons, transform=[[3]] * 200, synapse=None)
    nengo.Connection(task, error_item.neurons, transform=[[-3]] * 200, synapse=None)

    nengo.Connection(learning, error_loc.neurons, transform=[[3]] * 400, synapse=None)
    nengo.Connection(task, error_loc.neurons, transform=[[-3]] * 400, synapse=None)

    scaled_recall_loc = nengo.Node(env_scale_node, size_in=2, size_out=2)
    nengo.Connection(recall_loc, scaled_recall_loc, function=surface_to_env)

    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # This is only relevant when using velocity commands from a controller
    nengo.Connection(vel_input, env[:2])

    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(desired_pos, pos_error[:2], transform=-1)

    # nengo.Connection(pos_error, vel_input, function=compute_velocity)

    # Add a delta in the desired direction to the current location (for teleport mode)
    # nengo.Connection(pos_error[:2], vel_input, function=compute_direction_delta)
    # nengo.Connection(recall_loc, vel_input, function=surface_to_env)

    # For holonomic control
    # nengo.Connection(pos_error[:2], vel_input, transform=-1)
    nengo.Connection(pos_error[:2], vel_input, function=compute_direction_delta)

    # A user specified command for the location of the agent
    # command = nengo.Node([0, 0])
    command = nengo.Node(size_in=2, size_out=2)  # This version is used with the experiment class


    # Used directly in teleport mode
    # nengo.Connection(command, env[:2])

    def control_structure(t, x):
        # action == EXPLORE
        if x[0] > .5:
            # desired_pos=command
            # working_loc=current_loc
            # working_item=current_item
            # task=0 (learning on)
            return np.concatenate([[x[3]], [x[4]], [x[7]], [x[8]], x[11:11 + D], [0]])
        # action == RECALL
        elif x[1] > .5:
            # desired_pos=command
            # working_loc=query_loc_scaled
            # working_item=query_item
            # task=1 (learning off)
            return np.concatenate([[x[3]], [x[4]], [x[9]], [x[10]], x[18:18 + D], [1]])
        # action == FIND
        elif x[2] > .5:
            # desired_pos=scaled_recall_loc
            # working_loc=query_loc_scaled
            # working_item=query_item
            # task=1 (learning off)
            # return np.concatenate([[x[5]], [x[6]], [x[9]], [x[10]], x[18:18+D], [1]])
            return np.concatenate([[x[5]], [x[6]], [x[7]], [x[8]], x[18:18 + D], [1]])
        # action == LEARN (real room value is given in this case)
        else:
            # desired_pos=command
            # working_loc=current_loc
            # working_item=current_item
            # task=0 (learning on)
            return np.concatenate([[x[3]], [x[4]], [x[7]], [x[8]], x[11:11 + D], [0]])


    bg_node = nengo.Node(control_structure, size_in=3 + 2 + 2 + 2 + 2 + D + D, size_out=2 + 2 + D + 1)

    nengo.Connection(bg_node[[0, 1]], desired_pos, synapse=None)
    nengo.Connection(bg_node[[2, 3]], working_loc, synapse=None)
    nengo.Connection(bg_node[4:4 + D], working_item, synapse=None)
    nengo.Connection(bg_node[11], task, synapse=None)

    nengo.Connection(model.action.output, bg_node[[0, 1, 2]], synapse=None)
    nengo.Connection(command, bg_node[[3, 4]], synapse=None)
    nengo.Connection(scaled_recall_loc, bg_node[[5, 6]], synapse=None)
    nengo.Connection(env[:2], bg_node[[7, 8]], synapse=None)
    nengo.Connection(query_location, bg_node[[9, 10]], synapse=None)  # NOTE: this is not scaled yet
    nengo.Connection(env[3:], bg_node[11:11 + D], synapse=None)
    nengo.Connection(model.query_item.output, bg_node[18:18 + D], synapse=None)

    plot_item = EncoderPlot(conn_in_item)
    plot_loc = EncoderPlot(conn_in_loc)

    nengo.Connection(recall_item, exp_cont)
    nengo.Connection(exp_cont[[0, 1]], command)
    nengo.Connection(exp_cont[[2, 3, 4]], model.action.input)
    nengo.Connection(exp_cont[5:], model.query_item.input)


def on_step(sim):
    plot_item.update(sim)
    plot_loc.update(sim)


if __name__ == "__main__":
    sim = nengo.Simulator(model)
    sim.run((len(items) + num_test_pairs + 1) * time_per_item)
