import nengo
import numpy as np
import argparse
from encoders import to_ssp, to_bound_ssp, to_hex_region_ssp, to_band_region_ssp
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors, get_axes
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.networks.ssp_cleanup import SpatialCleanup
from ssp_navigation.utils.encodings import hilbert_2d


parser = argparse.ArgumentParser('Experiment comparing different encoder types for velocity controlled SSP movement')

parser.add_argument('--dim', default=512)
parser.add_argument('--limit', default=5)
parser.add_argument('--res', default=256)
parser.add_argument('--n-cconv-neurons', default=10)
parser.add_argument('--neurons-per-dim', default=5)
parser.add_argument('--encoder-type', default='place_cell', choices=['default', 'place_cell', 'band', 'grid_cell'])
parser.add_argument('--spacing', default=4, help='spacing for grid cell encoders')  # TODO: allow this to vary
parser.add_argument('--intercepts', default=0.2)
parser.add_argument(
    '--cleanup-path',
    default='/home/ctnuser/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/model.pt'
)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args([])

neurons_per_dim = 5
n_neurons = args.dim * args.neurons_per_dim

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)


# X, Y = get_axes(dim=dim, n=3, seed=13, period=0, optimal_phi=False)

# temporarily using these to match the axis vectors the cleanup was trained on
rstate = np.random.RandomState(seed=13)
X = make_good_unitary(args.dim, rng=rstate)
Y = make_good_unitary(args.dim, rng=rstate)


def angle_to_ssp(x):

    return encode_point(np.cos(x), np.sin(x), X, Y).v


heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)

rng = np.random.RandomState(seed=13)
preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)

encoders_place_cell = np.zeros((n_neurons, args.dim))
encoders_band_cell = np.zeros((n_neurons, args.dim))
encoders_grid_cell = np.zeros((n_neurons, args.dim))
for n in range(n_neurons):
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :], X=X, Y=Y)
    encoders_band_cell[n, :] = to_band_region_ssp(preferred_locations[n, :], X=X, Y=Y, angle=rng.uniform(0, 2*np.pi))
    encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], X=X, Y=Y, spacing=args.spacing)


spatial_heatmap = SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None)

model = nengo.Network(seed=13)
# model = spa.SPA(seed=13)
with model:
    # Initial kick to get the starting location SSP to represent 0,0
    initial_stim = nengo.Node(lambda t: 1 if t < 0.01 else 0)

    direction_heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
        size_in=args.dim, size_out=0,
    )
    location_heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
        size_in=args.dim, size_out=0,
    )

    # SSP representing the direction to move
    # model.direction_ssp = spa.State(args.dim)
    direction_ssp = nengo.Ensemble(n_neurons=n_neurons, dimensions=args.dim)

    # SSP representing the current location
    # model.location_ssp = spa.State(dim, feedback=1)
    # model.location_ssp = spa.State(args.dim)
    location_ssp = nengo.Ensemble(n_neurons=n_neurons, dimensions=args.dim)

    # Cleanup a potentially noisy spatial semantic pointer to a clean version
    spatial_cleanup = nengo.Node(
        SpatialCleanup(model_path=args.cleanup_path, dim=args.dim, hidden_size=512),
        size_in=args.dim, size_out=args.dim
    )

    # Circular convolution
    cconv = nengo.networks.CircularConvolution(args.n_cconv_neurons, dimensions=args.dim, invert_a=False, invert_b=False)
    nengo.Connection(location_ssp, cconv.input_a)
    nengo.Connection(direction_ssp, cconv.input_b)
    nengo.Connection(cconv.output, spatial_cleanup)
    nengo.Connection(spatial_cleanup, location_ssp)


    # nengo.Connection(model.location_ssp.output, cconv.input_a)
    # nengo.Connection(model.direction_ssp.output, cconv.input_b)
    # nengo.Connection(cconv.output, spatial_cleanup)
    # nengo.Connection(spatial_cleanup, model.location_ssp.input)

    # direction slider, for debugging
    direction_node = nengo.Node([0])

    # Convert chosen angle into a SSP corresponding to a point on a unit circle
    nengo.Connection(direction_node, direction_ssp, function=angle_to_ssp)

    # For the location (0, 0), the vector is just a 1 in the first position, and zeros elsewhere
    nengo.Connection(initial_stim, location_ssp[0])

    nengo.Connection(direction_ssp, direction_heatmap_node)
    nengo.Connection(location_ssp, location_heatmap_node)

    if args.encoder_type == 'default':
        direction_ssp.encoders = nengo.dists.UniformHypersphere(surface=True)
        location_ssp.encoders = nengo.dists.UniformHypersphere(surface=True)
    elif args.encoder_type == 'place_cell':
        direction_ssp.encoders = encoders_place_cell
        location_ssp.encoders = encoders_place_cell
    elif args.encoder_type == 'band':
        direction_ssp.encoders = encoders_band_cell
        location_ssp.encoders = encoders_band_cell
    elif args.encoder_type == 'grid_cell':
        direction_ssp.encoders = encoders_grid_cell
        location_ssp.encoders = encoders_grid_cell

    if args.encoder_type != 'default':
        direction_ssp.eval_points = encoders_place_cell
        # direction_ssp.intercepts = [args.intercepts]*n_neurons

        location_ssp.eval_points = encoders_place_cell
        # location_ssp.intercepts = [args.intercepts] * n_neurons
