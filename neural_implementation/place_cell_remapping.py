import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, get_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d

# generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp, normalize=True)

dim = 64#256

limit_low = -5
limit_high = 5
pc_limit_high = 30
pc_limit_low = -30
pc_limit_high = 5
pc_limit_low = -5
intercepts = .2

rng = np.random.RandomState(seed=13)
X, Y = get_axes(dim=dim, n=3, seed=13, period=0, optimal_phi=False)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v

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
dt = 0.001
n_samples = int((duration/2) / dt)

# set of positions to visit on a space filling curve, one for each timestep
# positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=8, N=2, normal_std=0)
positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=6, N=2, normal_std=0)

neurons_per_dim = 5
n_neurons = dim * neurons_per_dim
n_cconv_neurons = neurons_per_dim * 2

preferred_locations = hilbert_2d(pc_limit_low, pc_limit_high, n_neurons, rng, p=8, N=2, normal_std=3)


# item_sp = nengo.spa.SemanticPointer(dim)
item_sp = make_good_unitary(dim)
# item_sp = encode_point(1, 1, X, Y)


def input_func(t):
    pos = positions[min(int(np.floor(t/dt)), n_samples - 1)]
    pos_ssp = encode_point(pos[0], pos[1], X, Y)
    # item = item_sp
    bound = item_sp * pos_ssp

    return np.concatenate([pos, pos_ssp.v, bound.v])

encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
for n in range(n_neurons):
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])
    encoders_band_cell[n, :] = band_region_ssp(preferred_locations[n, :], angle=rng.uniform(0, 2*np.pi))
    encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], spacing=spacing)


model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
# model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    input_node = nengo.Node(
        input_func,
        size_in=0,
        size_out=2 + dim*2
    )
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)
    pos_2d = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

    memory = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)

    if encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
        memory.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
        memory.encoders = encoders_place_cell
    elif encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
        memory.encoders = encoders_band_cell
    elif encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell
        memory.encoders = encoders_grid_cell

    ssp_loc.eval_points = encoders_place_cell
    ssp_loc.intercepts = [intercepts]*n_neurons
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    memory.eval_points = encoders_place_cell
    ssp_loc.intercepts = [intercepts] * n_neurons
    # memory.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(input_node[2:2+dim], ssp_loc, synapse=None)
    nengo.Connection(input_node[2 + dim:2 + dim*2], memory, synapse=None)
    nengo.Connection(input_node[:2], pos_2d, synapse=None)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
        bound_spikes_p = nengo.Probe(memory.neurons, synapse=0.01)
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
        nengo.Connection(memory, memory_heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(duration)

    # fname = 'output_pc_remap_difflimit_{}_{}s.npz'.format(encoder_type, duration)
    fname = 'output_pc_remap_intercept_dim{}_{}_{}s.npz'.format(dim, encoder_type, duration)

    np.savez(
        fname,
        spikes=sim.data[spikes_p],
        bound_spikes=sim.data[bound_spikes_p],
        pos_2d=sim.data[pos_2d_p],
    )
