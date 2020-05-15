import nengo
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_axes, generate_region_vector, \
    get_heatmap_vectors, encode_point_hex
from spatial_semantic_pointers.plots import SpatialHeatmap
from ssp_navigation.utils.encodings import hilbert_2d
import argparse
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir, orthogonal_hex_dir_7dim

parser = argparse.ArgumentParser("Gather data on spatial activations of neurons with different encoders")

# parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5.)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--encoder-type', type=str, default='grid_cell',
                    choices=['place_cell', 'grid_cell', 'band', 'default'])
parser.add_argument('--duration', type=int, default=70)
parser.add_argument('--neurons-per-dim', type=int, default=5)
parser.add_argument('--intercepts', type=float, default=0.2)

args = parser.parse_args()


rng = np.random.RandomState(seed=args.seed)

phis = (np.pi / 2., np.pi/3., np.pi/5.)
angles = (0, np.pi/3., np.pi/5.)
X, Y = orthogonal_hex_dir(phis=phis, angles=angles)
dim = len(X.v)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


model = nengo.Network(seed=args.seed)


dt = 0.001
n_samples = int(args.duration / dt)

# set of positions to visit on a space filling curve, one for each timestep
positions = hilbert_2d(-args.limit, args.limit, n_samples, rng, p=8, N=2, normal_std=0)

n_neurons = dim * args.neurons_per_dim

preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)


encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
for n in range(n_neurons):
    ind = rng.randint(0, len(phis))
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])

    encoders_grid_cell[n, :] = grid_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind
    )

    encoders_band_cell[n, :] = band_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind,
        band_index=rng.randint(0, 3)
    )

# model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
model.config[nengo.Ensemble].neuron_type=nengo.LIF()
with model:
    pos_2d = nengo.Node(lambda t: positions[min(int(np.floor(t/dt)), n_samples - 1)], size_in=0, size_out=2)
    ssp_loc = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)

    if args.encoder_type == 'default':
        ssp_loc.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif args.encoder_type == 'place_cell':
        ssp_loc.encoders = encoders_place_cell
    elif args.encoder_type == 'band':
        ssp_loc.encoders = encoders_band_cell
    elif args.encoder_type == 'grid_cell':
        ssp_loc.encoders = encoders_grid_cell
    # TODO: add mixed encoder option

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
            size_in=dim,
            size_out=0,
        )

        nengo.Connection(ssp_loc, heatmap_node)

if __name__ == '__main__':
    sim = nengo.Simulator(model, dt=dt)
    sim.run(args.duration)

    folder = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output'
    fname = '{}/output_orth_hex_toroid_d{}_{}_{}s.npz'.format(folder, dim, args.encoder_type, args.duration)
    print("saving to: {}".format(fname))
    np.savez(
        fname,
        spikes=sim.data[spikes_p],
        pos_2d=sim.data[pos_2d_p],
    )
