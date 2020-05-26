# Moving an SSP representation of current position around a map
import nengo
import nengo.spa as spa
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.networks.ssp_cleanup import SpatialCleanup

seed = 13
dim = 512
limit = 5
res = 256

model = spa.SPA(seed=seed)

rstate = np.random.RandomState(seed=13)

x_axis_sp = make_good_unitary(dim, rng=rstate)
y_axis_sp = make_good_unitary(dim, rng=rstate)

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

n_cconv_neurons = 50#10

#ssp_cleanup_path = '/home/bjkomer/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/model.pt'
ssp_cleanup_path = '/home/ctnuser/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/model.pt'


def angle_to_ssp(x):

    return encode_point(np.cos(x), np.sin(x), x_axis_sp, y_axis_sp).v


heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

spatial_heatmap = SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None)

with model:

    # Initial kick to get the starting location SSP to represent 0,0
    initial_stim = nengo.Node(lambda t: 1 if t < 0.01 else 0)

    direction_heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
        size_in=dim, size_out=0,
    )
    location_heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
        size_in=dim, size_out=0,
    )

    # SSP representing the direction to move
    model.direction_ssp = spa.State(dim)

    # SSP representing the current location
    # model.location_ssp = spa.State(dim, feedback=1)
    model.location_ssp = spa.State(dim)

    # Cleanup a potentially noisy spatial semantic pointer to a clean version
    spatial_cleanup = nengo.Node(
        SpatialCleanup(model_path=ssp_cleanup_path, dim=dim, hidden_size=512),
        size_in=dim, size_out=dim
    )

    # Circular convolution
    cconv = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=dim, invert_a=False, invert_b=False)
    nengo.Connection(model.location_ssp.output, cconv.input_a)
    nengo.Connection(model.direction_ssp.output, cconv.input_b)
    nengo.Connection(cconv.output, spatial_cleanup)
    nengo.Connection(spatial_cleanup, model.location_ssp.input)

    # SSP representing the map with items
    # TODO: can just have this a fixed node, it doesn't change
    model.map_ssp = spa.State(dim)

    # direction slider, for debugging
    direction_node = nengo.Node([0])

    # Convert chosen angle into a SSP corresponding to a point on a unit circle
    nengo.Connection(direction_node, model.direction_ssp.input, function=angle_to_ssp)

    # For the location (0, 0), the vector is just a 1 in the first position, and zeros elsewhere
    nengo.Connection(initial_stim, model.location_ssp.input[0])

    nengo.Connection(model.direction_ssp.output, direction_heatmap_node)
    nengo.Connection(model.location_ssp.output, location_heatmap_node)
