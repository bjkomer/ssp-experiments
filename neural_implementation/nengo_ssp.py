import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, get_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d

# generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp, normalize=True)

dim = 256

limit_low = -5
limit_high = 5

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
][0]


model = nengo.Network(seed=13)


duration = 70
dt = 0.001
n_samples = int(duration / dt)

# set of positions to visit on a space filling curve, one for each timestep
positions = hilbert_2d(limit_low, limit_high, n_samples, rng, p=8, N=2, normal_std=0)

n_neurons = dim * 5  # * 5

preferred_locations = hilbert_2d(limit_low, limit_high, n_neurons, rng, p=8, N=2, normal_std=3)




encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
for n in range(n_neurons):
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])
    encoders_band_cell[n, :] = band_region_ssp(preferred_locations[n, :], angle=rng.uniform(0, 2*np.pi))
    encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], spacing=spacing)


# model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    pos_2d = nengo.Node(lambda t: positions[min(int(np.floor(t/dt)), n_samples - 1)], size_in=0, size_out=2)
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)

    if encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
    elif encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
    elif encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell

    ssp_loc.eval_points = encoders_place_cell
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(pos_2d, ssp_loc, function=to_ssp)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
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

        nengo.Connection(ssp_loc, heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(duration)

    fname = 'output_{}_{}s.npz'.format(encoder_type, duration)

    np.savez(
        fname,
        spikes=sim.data[spikes_p],
        pos_2d=sim.data[pos_2d_p],
    )
