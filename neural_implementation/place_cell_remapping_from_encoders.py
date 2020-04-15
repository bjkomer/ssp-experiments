import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, get_heatmap_vectors, get_fixed_dim_grid_axes
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d
import os
import argparse

# TODO: have only one population, and run things twice, ensuring the same encoders are used
#       one run through will have just locations, the next will be convolved with some SP
#       the SP that it is convolved with may need to be dependent on the location (sensors?)

parser = argparse.ArgumentParser('Experiment with place cell remapping based on encoders')

parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--n-envs', type=int, default=10)
# parser.add_argument('--prob-pc', type=float, default=.7,
#                     help='probability that a neuron appears as a place cell in a given environment')
parser.add_argument('--limit', type=float, default=5.)
parser.add_argument('--intercepts', type=float, default=0.2)
parser.add_argument('--res', type=int, default=32, help='resolution of the output images')
args = parser.parse_args()

path_prefix = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output'

# if not os.path.exists('output'):
#     os.makedirs('output')

diff_axis = True
grid_axes = False


rng = np.random.RandomState(seed=13)

Xs = []
Ys = []
for i in range(args.n_envs):
    X, Y = get_axes(dim=args.dim, n=3, seed=13+i, period=0, optimal_phi=False)
    Xs.append(X)
    Ys.append(Y)


# if grid_axes:
#     X, Y = get_fixed_dim_grid_axes(dim=args.dim, seed=13)
# else:
#     X, Y = get_axes(dim=args.dim, n=3, seed=13, period=0, optimal_phi=False)
# # X_new, Y_new = get_axes(dim=dim, n=3, seed=14, period=0, optimal_phi=False)



def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


def to_bound_ssp(v, item):

    return (item * encode_point(v[0], v[1], X, Y)).v

# 3 directions 120 degrees apart
vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]
spacing = 4


def to_hex_region_ssp(v, spacing=4):

    ret = np.zeros((args.dim,))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for i in range(3):
        ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v

    return ret


def band_region_ssp(v, angle):

    ret = np.zeros((args.dim,))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for dx in np.linspace(20./63., 20, 64):
        ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v

    return ret




model = nengo.Network(seed=13)


# duration = 70
# one pass for each environment
duration = 40*args.n_envs
dt = 0.001
n_samples = int((duration/args.n_envs) / dt)

# set of positions to visit on a space filling curve, one for each timestep
# positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=8, N=2, normal_std=0)
positions = hilbert_2d(-args.limit, args.limit, n_samples, rng, p=6, N=2, normal_std=0)

neurons_per_dim = 5
n_neurons = args.dim * neurons_per_dim
n_cconv_neurons = neurons_per_dim * 2

# preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)
# preferred_locations = hilbert_2d(pc_limit_low, pc_limit_high, n_neurons, rng, p=10, N=2, normal_std=3)

# preferred locations for each neurons, for each environment
# using 2x limit so there is a chance that the neuron does not have a place field in the environment
preferred_locations = rng.uniform(-2*args.limit, 2*args.limit, (args.n_envs, n_neurons, 2))


def input_func(t):
    index = int(np.floor(t/dt))
    pos = positions[index % (n_samples - 1)]
    env_index = (index // (n_samples - 1)) % args.n_envs
    pos_ssp = encode_point(pos[0], pos[1], Xs[env_index], Ys[env_index])

    return np.concatenate([pos, pos_ssp.v])


encoders_multi_place_cell = np.zeros((n_neurons, args.dim))

for n in range(n_neurons):
    for i in range(args.n_envs):
        encoders_multi_place_cell[n, :] += encode_point(
            preferred_locations[i, n, 0], preferred_locations[i, n, 1], Xs[i], Ys[i]
        ).v
    # normalize
    encoders_multi_place_cell[n, :] /= np.linalg.norm(encoders_multi_place_cell[n, :])



model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
# model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    input_node = nengo.Node(
        input_func,
        size_in=0,
        size_out=2 + args.dim
    )
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=args.dim)
    pos_2d = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

    ssp_loc.encoders = encoders_multi_place_cell
    ssp_loc.eval_points = encoders_multi_place_cell
    ssp_loc.intercepts = [args.intercepts]*n_neurons
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(input_node[2:2+args.dim], ssp_loc, synapse=None)
    nengo.Connection(input_node[:2], pos_2d, synapse=None)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
        # bound_spikes_p = nengo.Probe(memory.neurons, synapse=0.01)
        pos_2d_p = nengo.Probe(pos_2d)
    else:
        assert False # this code is still old
        res = 128
        xs = np.linspace(-args.limit, args.limit, res)
        ys = np.linspace(-args.limit, args.limit, res)
        heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
        # create a visualization of the heatmap
        heatmap_node = nengo.Node(
            SpatialHeatmap(
                heatmap_vectors,
                xs,
                ys,
                cmap='plasma',
                vmin=-1,  # None,
                vmax=1,  # None,
            ),
            size_in=args.dim,
            size_out=0,
        )

        memory_heatmap_node = nengo.Node(
            SpatialHeatmap(
                heatmap_vectors,
                xs,
                ys,
                cmap='plasma',
                vmin=-1,  # None,
                vmax=1,  # None,
            ),
            size_in=args.dim,
            size_out=0,
        )

        nengo.Connection(ssp_loc, heatmap_node)
        # nengo.Connection(memory, memory_heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(duration)


    fname = '{}/pc_remap_encoders_dim{}_limit{}_{}s.npz'.format(path_prefix, args.dim, args.limit, duration)

    # np.savez(
    #     fname,
    #     spikes=sim.data[spikes_p][:int((duration/dt)/2)],
    #     bound_spikes=sim.data[spikes_p][int((duration/dt)/2):],
    #     pos_2d=sim.data[pos_2d_p][:int((duration/dt)/2)],
    # )

    # doing processing before saving to save space on disk

    pos = sim.data[pos_2d_p][:int((duration/dt)/args.n_envs)]
    data_per_env = int((duration / dt) / args.n_envs)
    spikes = np.zeros((args.n_envs, data_per_env, n_neurons))
    for i in range(args.n_envs):
        spikes[i, :, :] = sim.data[spikes_p][data_per_env*i:data_per_env*(i+1), :]

    xs = np.linspace(-args.limit, args.limit, args.res)
    ys = np.linspace(-args.limit, args.limit, args.res)
    diff = xs[1] - xs[0]

    # spatial firing for each neuron
    img = np.zeros((args.n_envs, n_neurons, args.res, args.res))

    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            ind = np.where((pos[:, 0] >= x) & (pos[:, 0] < x + diff) & (pos[:, 1] >= y) & (pos[:, 1] < y + diff))
            if len(ind[0]) > 0:
                for ei in range(args.n_envs):
                    img[ei, :, i, j] = np.mean(spikes[ei, ind[0], :], axis=0)

    np.savez(
        fname,
        img=img,
    )

    # np.savez(
    #     fname,
    #     spikes=sim.data[spikes_p],
    #     bound_spikes=sim.data[bound_spikes_p],
    #     pos_2d=sim.data[pos_2d_p],
    # )
