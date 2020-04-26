import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, \
    get_heatmap_vectors, encode_point_hex
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d
import argparse

parser = argparse.ArgumentParser("Gather data on spatial activations of neurons with different encoders")

parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5.)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--encoder-type', type=str, default='place_cell',
                    choices=['place_cell', 'grid_cell', 'band', 'default'])
parser.add_argument('--duration', type=int, default=70)
parser.add_argument('--neurons-per-dim', type=int, default=5)
parser.add_argument('--intercepts', type=float, default=0.2)

args = parser.parse_args()

spacings = [3, 4, 5]


rng = np.random.RandomState(seed=args.seed)
# X, Y = get_axes(dim=dim, n=3, seed=13, period=0, optimal_phi=False)
X = make_good_unitary(dim=args.dim, rng=rng)
Y = make_good_unitary(dim=args.dim, rng=rng)
Z = make_good_unitary(dim=args.dim, rng=rng)


def to_ssp(v):

    # return encode_point(v[0], v[1], X, Y).v
    return encode_point_hex(v[0], v[1], X, Y, Z).v


# 3 directions 120 degrees apart
base_vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]

def to_hex_region_ssp(v, spacing=4, angle=0.0):

    ret = np.zeros((args.dim,))
    vec_dirs = np.zeros((3, ))
    for i in range(3):
        vec_dirs[i] = base_vec_dirs[i] + angle
    ret[:] = encode_point_hex(v[0], v[1], X, Y, Z).v
    for i in range(3):
        # ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        # ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point_hex(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y, Z).v
        ret += encode_point_hex(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y, Z).v

    return ret


# def to_hex_region_ssp(v, spacing=4, angle=0.0):
#
#     ret = np.zeros((args.dim,))
#     vec_dirs = np.zeros((3, ))
#     for i in range(3):
#         vec_dirs[i] = base_vec_dirs[i] + angle
#     # ret[:] = encode_point(v[0], v[1], X, Y).v
#     # ret[:] = encode_point_hex(v[0], v[1], X, Y, Z).v
#     for i in range(2):  # just two to make a sloped grid
#         for xdist in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
#             for ydist in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
#                 # ret += encode_point(v[0] + xdist * spacing * np.cos(vec_dirs[i]), v[1] + ydist * spacing * np.sin(vec_dirs[i]), X, Y).v
#                 ret += encode_point_hex(
#                     v[0] + xdist * spacing * np.cos(vec_dirs[i]),
#                     v[1] + ydist * spacing * np.sin(vec_dirs[i]),
#                     X, Y, Z
#                 ).v
#
#     return ret


def band_region_ssp(v, angle):

    ret = np.zeros((args.dim,))
    # ret[:] = encode_point(v[0], v[1], X, Y).v
    ret[:] = encode_point_hex(v[0], v[1], X, Y, Z).v
    for dx in np.linspace(20./63., 20, 64):
        # ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        # ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v
        ret += encode_point_hex(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y, Z).v
        ret += encode_point_hex(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y, Z).v

    return ret


model = nengo.Network(seed=args.seed)


dt = 0.001
n_samples = int(args.duration / dt)

# set of positions to visit on a space filling curve, one for each timestep
positions = hilbert_2d(-args.limit, args.limit, n_samples, rng, p=8, N=2, normal_std=0)

n_neurons = args.dim * args.neurons_per_dim

preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)


encoders_place_cell = np.zeros((n_neurons, args.dim))
encoders_band_cell = np.zeros((n_neurons, args.dim))
encoders_grid_cell = np.zeros((n_neurons, args.dim))
for n in range(n_neurons):
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])
    encoders_band_cell[n, :] = band_region_ssp(preferred_locations[n, :], angle=rng.uniform(0, 2*np.pi))
    spacing = rng.choice(spacings)
    angle = rng.uniform(0, 2*np.pi)
    encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], spacing=spacing, angle=angle)


# model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    pos_2d = nengo.Node(lambda t: positions[min(int(np.floor(t/dt)), n_samples - 1)], size_in=0, size_out=2)
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=args.dim)

    if args.encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif args.encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
    elif args.encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
    elif args.encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell

    ssp_loc.eval_points = encoders_place_cell
    ssp_loc.intercepts = [args.intercepts]*n_neurons
    # ssp_loc.eval_points = nengo.dists.UniformHypersphere(surface=True)

    nengo.Connection(pos_2d, ssp_loc, function=to_ssp)

    if __name__ == '__main__':
        # probes
        spikes_p = nengo.Probe(ssp_loc.neurons, synapse=0.01)
        pos_2d_p = nengo.Probe(pos_2d)
    else:
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

        nengo.Connection(ssp_loc, heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(args.duration)

    folder = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output'
    fname = '{}/output_d{}_{}_{}s.npz'.format(folder, args.dim, args.encoder_type, args.duration)

    np.savez(
        fname,
        spikes=sim.data[spikes_p],
        pos_2d=sim.data[pos_2d_p],
    )
