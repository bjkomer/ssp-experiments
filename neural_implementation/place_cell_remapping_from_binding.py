import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, get_heatmap_vectors, get_fixed_dim_grid_axes
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d
import os

# TODO: have only one population, and run things twice, ensuring the same encoders are used
#       one run through will have just locations, the next will be convolved with some SP
#       the SP that it is convolved with may need to be dependent on the location (sensors?)
# assert False # TODO: add the above

# generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp, normalize=True)

path_prefix = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output'

# if not os.path.exists('output'):
#     os.makedirs('output')

diff_axis = True
grid_axes = True

dim = 512#256#64#256

# limit_high = 30
# limit_low = -30
limit_high = 5
limit_low = -5
# pc_limit_high = 30
# pc_limit_low = -30
pc_limit_low = -5
pc_limit_high = 5
intercepts = .2

rng = np.random.RandomState(seed=13)
if grid_axes:
    X, Y = get_fixed_dim_grid_axes(dim=dim, seed=13)
else:
    X, Y = get_axes(dim=dim, n=3, seed=13, period=0, optimal_phi=False)
# X_new, Y_new = get_axes(dim=dim, n=3, seed=14, period=0, optimal_phi=False)



def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


def to_bound_ssp(v, item):

    return (item * encode_point(v[0], v[1], X, Y)).v

# 3 directions 120 degrees apart
vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]
spacing = 4


def to_hex_region_ssp(v, spacing=4):

    ret = np.zeros((dim,))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for i in range(3):
        ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v

    return ret


def band_region_ssp(v, angle):

    ret = np.zeros((dim,))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for dx in np.linspace(20./63., 20, 64):
        ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v

    return ret





encoder_type = [
    'default',
    'place_cell',
    'band',
    'grid_cell'
][1]


model = nengo.Network(seed=13)


# duration = 70
# one pass for one 'environment' another pass for a different one
duration = 40*2
# duration = 80*2
# duration = 160*2
# duration = 200*2
dt = 0.001
n_samples = int((duration/2) / dt)

# set of positions to visit on a space filling curve, one for each timestep
# positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=8, N=2, normal_std=0)
positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=6, N=2, normal_std=0)

neurons_per_dim = 5
n_neurons = dim * neurons_per_dim
n_cconv_neurons = neurons_per_dim * 2

preferred_locations = hilbert_2d(pc_limit_low, pc_limit_high, n_neurons, rng, p=8, N=2, normal_std=3)
# preferred_locations = hilbert_2d(pc_limit_low, pc_limit_high, n_neurons, rng, p=10, N=2, normal_std=3)


# item_sp = nengo.spa.SemanticPointer(dim)
item_sp = make_good_unitary(dim, rng=rng)
item_sp_2 = make_good_unitary(dim, rng=rng)
# item_sp = encode_point(1, 1, X, Y)


def input_func(t):
    index = int(np.floor(t/dt))
    pos = positions[index % (n_samples - 1)]
    pos_ssp = encode_point(pos[0], pos[1], X, Y)
    # item = item_sp

    bound_first = item_sp * pos_ssp
    bound_second = item_sp_2 * pos_ssp

    if index > n_samples - 1:
        return np.concatenate([pos, bound_second.v])
    else:
        return np.concatenate([pos, bound_first.v])

encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
for n in range(n_neurons):
    encoders_place_cell[n, :] = to_bound_ssp(preferred_locations[n, :], item_sp)
    encoders_band_cell[n, :] = band_region_ssp(preferred_locations[n, :], angle=rng.uniform(0, 2*np.pi))
    encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], spacing=spacing)


model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
# model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    input_node = nengo.Node(
        input_func,
        size_in=0,
        size_out=2 + dim
    )
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)
    pos_2d = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

    # memory = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)

    if encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
        # memory.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
        # memory.encoders = encoders_place_cell
    elif encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
        # memory.encoders = encoders_band_cell
    elif encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell
        # memory.encoders = encoders_grid_cell

    ssp_loc.eval_points = encoders_place_cell
    ssp_loc.intercepts = [intercepts]*n_neurons
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    # memory.eval_points = encoders_place_cell
    ssp_loc.intercepts = [intercepts] * n_neurons
    # memory.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(input_node[2:2+dim], ssp_loc, synapse=None)
    # nengo.Connection(input_node[2 + dim:2 + dim*2], memory, synapse=None)
    nengo.Connection(input_node[:2], pos_2d, synapse=None)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
        # bound_spikes_p = nengo.Probe(memory.neurons, synapse=0.01)
        pos_2d_p = nengo.Probe(pos_2d)
    else:
        res = 128
        xs = np.linspace(limit_low, limit_high, res)
        ys = np.linspace(limit_low, limit_high, res)
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
            size_in=dim,
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
            size_in=dim,
            size_out=0,
        )

        nengo.Connection(ssp_loc, heatmap_node)
        # nengo.Connection(memory, memory_heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(duration)


    fname = '{}/pc_remap_bound_dim{}_limit{}_{}_{}s.npz'.format(path_prefix, dim, limit_high, encoder_type, duration)

    # np.savez(
    #     fname,
    #     spikes=sim.data[spikes_p][:int((duration/dt)/2)],
    #     bound_spikes=sim.data[spikes_p][int((duration/dt)/2):],
    #     pos_2d=sim.data[pos_2d_p][:int((duration/dt)/2)],
    # )

    # doing processing before saving to save space on disk

    pos = sim.data[pos_2d_p][:int((duration/dt)/2)]
    spikes = sim.data[spikes_p][:int((duration / dt) / 2)]
    bound_spikes = sim.data[spikes_p][int((duration / dt) / 2):]

    res = 32#64#128
    xs = np.linspace(limit_low, limit_high, res)
    ys = np.linspace(limit_low, limit_high, res)
    diff = xs[1] - xs[0]

    # spatial firing for each neuron
    img = np.zeros((n_neurons, res, res))
    bound_img = np.zeros((n_neurons, res, res))

    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            ind = np.where((pos[:, 0] >= x) & (pos[:, 0] < x + diff) & (pos[:, 1] >= y) & (pos[:, 1] < y + diff))
            if len(ind[0]) > 0:
                img[:, i, j] = np.mean(spikes[ind[0], :], axis=0)
                bound_img[:, i, j] = np.mean(bound_spikes[ind[0], :], axis=0)

    np.savez(
        fname,
        img=img,
        bound_img=bound_img,
    )

    # np.savez(
    #     fname,
    #     spikes=sim.data[spikes_p],
    #     bound_spikes=sim.data[bound_spikes_p],
    #     pos_2d=sim.data[pos_2d_p],
    # )
