# Moving an SSP representation of current position around a map
import nengo
import nengo.spa as spa
# import nengo_spa as spa
import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.networks.ssp_cleanup import SpatialCleanup
from ssp_navigation.utils.encodings import hilbert_2d, get_encoding_function, add_encoding_params
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir
import os
import argparse

parser = argparse.ArgumentParser('Path integration with nengo and SSPs')
parser = add_encoding_params(parser)
args = parser.parse_args([])

use_spa = True

args.spatial_encoding = 'sub-toroid-ssp'
args.spatial_encoding = 'hex-ssp'
args.spatial_encoding = 'ssp'
# args.dim = 256
args.dim = 512

limit_low = 0
limit_high = 13

seed = 13
limit = 10#5
res = 256
neurons_per_dim = 10#50

enc_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

X = spa.SemanticPointer(data=enc_func(1, 0))
Y = spa.SemanticPointer(data=enc_func(0, 1))

# get exact params of the encoding used by using the same seed
eps=0.001
n_toroids = ((args.dim - 1) // 2) // args.n_proj
rng_params = np.random.RandomState(seed=args.seed)
# Randomly select the angle for each toroid
phis = rng_params.uniform(0, 2 * np.pi, size=(n_toroids,))
angles = rng_params.uniform(-np.pi + eps, np.pi - eps, size=n_toroids)

rng = np.random.RandomState(seed=seed)

# phis = (np.pi*.75, np.pi / 2., np.pi/3., np.pi/5., np.pi*.4, np.pi*.6)
# # angles = rng.uniform(0, 2*np.pi, size=len(phis))#(0, np.pi/3., np.pi/5.)
# angles = (0, np.pi*.3, np.pi*.2, np.pi*.4, np.pi*.1, np.pi*.5)
# X, Y = orthogonal_hex_dir(phis=phis, angles=angles)
dim = len(X.v)

# xs = np.linspace(-limit, limit, res)
# ys = np.linspace(-limit, limit, res)
xs = np.linspace(limit_low-1, limit_high+1, res)
ys = np.linspace(limit_low-1, limit_high+1, res)

n_cconv_neurons = 10#50#10

n_neurons = neurons_per_dim * dim

# ssp_cleanup_path = '/home/ctnuser/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/model.pt'
# ssp_cleanup_path = '/home/ctnuser/ssp_navigation_sandbox/neural_implementation/trained_models/ssp_cleanup/May19_02-51-46/model.pt'
if args.spatial_encoding == 'sub-toroid-ssp':
    if args.dim == 256:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/sub-toroid-ssp/d256/May23_13-36-51/model.pt'
    elif args.dim == 512:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/sub-toroid-ssp/d512/May23_14-16-04/model.pt'
    elif args.dim == 1024:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/sub-toroid-ssp/d1024/May23_15-29-12/model.pt'
elif args.spatial_encoding == 'hex-ssp':
    if args.dim == 256:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/hex-ssp/d256/May23_13-54-52/model.pt'
    elif args.dim == 512:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/hex-ssp/d512/May23_14-48-23/model.pt'
    elif args.dim == 1024:
        ssp_cleanup_path = '/home/ctnuser/ssp-navigation/ssp_navigation/trained_models/ssp_cleanup/hex-ssp/d1024/May23_16-11-12/model.pt'
else:
    ssp_cleanup_path = '/home/ctnuser/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/model.pt'
    seed = 13
    dim = 512
    limit = 5
    res = 256

    rstate = np.random.RandomState(seed=13)

    X = make_good_unitary(dim, rng=rstate)
    Y = make_good_unitary(dim, rng=rstate)

    xs = np.linspace(-limit, limit, res)
    ys = np.linspace(-limit, limit, res)

pytorch_cleanup = True

if not pytorch_cleanup:
    # generate a dataset for the cleanup function
    cache_fname = 'cleanup_dataset_{}.npz'.format(dim)
    if os.path.exists(cache_fname):
        data = np.load(cache_fname)
        clean_vectors = data['clean_vectors']
        noisy_vectors = data['noisy_vectors']
    else:
        n_samples = 50000
        clean_vectors = np.zeros((n_samples, dim))
        noisy_vectors = np.zeros((n_samples, dim))
        # locations = rng.uniform(-limit, limit, size=(n_samples, 2))
        locations = rng.uniform(limit_low, limit_high, size=(n_samples, 2))
        noise = rng.normal(0, 0.5/np.sqrt(dim), size=(n_samples, dim))
        for i in range(n_samples):
            clean_vectors[i, :] = encode_point(locations[i, 0], locations[i, 1], X, Y).v
            noisy_vectors[i, :] = clean_vectors[i, :] + noise[i, :]
        np.savez(cache_fname, clean_vectors=clean_vectors, noisy_vectors=noisy_vectors)


def angle_to_ssp(x):

    return encode_point(np.cos(x), np.sin(x), X, Y).v


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)

# for direction SSP visualization
xs_dir = np.linspace(-1.5, 1.5, 32)
ys_dir = np.linspace(-1.5, 1.5, 32)
heatmap_vectors_dir = get_heatmap_vectors(xs_dir, ys_dir, X, Y)

if not use_spa:

    # spatial_heatmap = SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None)

    # preferred_locations = hilbert_2d(-limit, limit, n_neurons, rng, p=8, N=2, normal_std=3)
    preferred_locations = hilbert_2d(limit_low, limit_high, n_neurons, rng, p=8, N=2, normal_std=3)

    encoders_place_cell = np.zeros((n_neurons, dim))
    encoders_band_cell = np.zeros((n_neurons, dim))
    encoders_grid_cell = np.zeros((n_neurons, dim))
    encoders_mixed = np.zeros((n_neurons, dim))
    mixed_intercepts = []
    for n in range(n_neurons):
        ind = rng.randint(0, len(phis))
        encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])

        encoders_grid_cell[n, :] = grid_cell_encoder(
            location=preferred_locations[n, :],
            dim=dim, phi=phis[ind], angle=angles[ind],
            toroid_index=ind
        )

        band_ind = rng.randint(0, 3)
        encoders_band_cell[n, :] = band_cell_encoder(
            location=preferred_locations[n, :],
            dim=dim, phi=phis[ind], angle=angles[ind],
            toroid_index=ind,
            band_index=band_ind
        )

        mix_ind = rng.randint(0, 3)
        if mix_ind == 0:
            encoders_mixed[n, :] = encoders_place_cell[n, :]
            mixed_intercepts.append(.3)
        elif mix_ind == 1:
            encoders_mixed[n, :] = encoders_grid_cell[n, :]
            mixed_intercepts.append(.2)
        elif mix_ind == 2:
            encoders_mixed[n, :] = encoders_band_cell[n, :]
            mixed_intercepts.append(0.)


    preferred_locations_dir = hilbert_2d(-1.5, 1.5, n_neurons, rng, p=8, N=2, normal_std=3)
    encoders_place_cell_dir = np.zeros((n_neurons, dim))
    for n in range(n_neurons):
        encoders_place_cell_dir[n, :] = to_ssp(preferred_locations_dir[n, :])

    model = nengo.Network(seed=seed)
    model.config[nengo.Ensemble].neuron_type=nengo.LIF()
    # model.config[nengo.Ensemble].neuron_type=nengo.Direct()
    with model:

        # Initial kick to get the starting location SSP to represent 0,0
        # initial_stim = nengo.Node(lambda t: 1 if t < 0.01 else 0)
        initial_stim = nengo.Node(lambda t: 1 if t < 0.1 else 0)

        direction_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors_dir, xs_dir, ys_dir, cmap='plasma', vmin=None, vmax=None),
            size_in=dim, size_out=0,
        )
        location_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
            size_in=dim, size_out=0,
        )

        # SSP representing the direction to move
        direction_ssp = nengo.Ensemble(n_neurons=dim*neurons_per_dim, dimensions=dim, neuron_type=nengo.LIF())

        # SSP representing the current location
        location_ssp = nengo.Ensemble(n_neurons=dim*neurons_per_dim, dimensions=dim, neuron_type=nengo.LIF())

        # direction_ssp.encoders = encoders_grid_cell
        # direction_ssp.eval_points = encoders_grid_cell
        # direction_ssp.encoders = encoders_mixed
        direction_ssp.encoders = encoders_place_cell_dir
        direction_ssp.eval_points = encoders_place_cell_dir
        # direction_ssp.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
        direction_ssp.intercepts = mixed_intercepts

        # location_ssp.encoders = encoders_grid_cell
        # location_ssp.encoders = encoders_mixed
        location_ssp.encoders = encoders_place_cell
        location_ssp.eval_points = encoders_place_cell
        # location_ssp.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
        location_ssp.intercepts = mixed_intercepts

        # Cleanup a potentially noisy spatial semantic pointer to a clean version
        if pytorch_cleanup:
            spatial_cleanup = nengo.Node(
                SpatialCleanup(model_path=ssp_cleanup_path, dim=args.dim, hidden_size=args.dim),
                size_in=args.dim, size_out=args.dim
            )
            nengo.Connection(spatial_cleanup, location_ssp)
        else:
            spatial_cleanup = nengo.Ensemble(n_neurons=dim*neurons_per_dim, dimensions=dim, neuron_type=nengo.LIF())
            spatial_cleanup.encoders = encoders_mixed
            spatial_cleanup.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
            spatial_cleanup.intercepts = mixed_intercepts
            nengo.Connection(
                spatial_cleanup,
                location_ssp,
                function=clean_vectors*1.1,
                eval_points=noisy_vectors,
                scale_eval_points=False,
            )

        # Circular convolution
        cconv = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=dim, invert_a=False, invert_b=False)
        nengo.Connection(location_ssp, cconv.input_a)
        nengo.Connection(direction_ssp, cconv.input_b)
        nengo.Connection(cconv.output, spatial_cleanup)


        # SSP representing the map with items
        # TODO: can just have this a fixed node, it doesn't change
        # model.map_ssp = spa.State(dim)

        # direction slider, for debugging
        direction_node = nengo.Node([0])

        # Convert chosen angle into a SSP corresponding to a point on a unit circle
        nengo.Connection(direction_node, direction_ssp, function=angle_to_ssp)

        # For the location (0, 0), the vector is just a 1 in the first position, and zeros elsewhere
        nengo.Connection(initial_stim, location_ssp[0])

        nengo.Connection(direction_ssp, direction_heatmap_node)
        nengo.Connection(location_ssp, location_heatmap_node)
else:
    model = spa.SPA(seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIF()
    # model.config[nengo.Ensemble].neuron_type=nengo.Direct()
    with model:

        # Initial kick to get the starting location SSP to represent 0,0
        # initial_stim = nengo.Node(lambda t: 1 if t < 0.01 else 0)
        initial_stim = nengo.Node(lambda t: 1 if t < 0.1 else 0)

        direction_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors_dir, xs_dir, ys_dir, cmap='plasma', vmin=None, vmax=None),
            size_in=dim, size_out=0,
        )
        location_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
            size_in=dim, size_out=0,
        )

        # SSP representing the direction to move
        model.direction_ssp = spa.State(dim)

        # SSP representing the current location
        model.location_ssp = spa.State(dim)

        # Cleanup a potentially noisy spatial semantic pointer to a clean version
        if pytorch_cleanup:
            spatial_cleanup = nengo.Node(
                SpatialCleanup(model_path=ssp_cleanup_path, dim=args.dim, hidden_size=args.dim),
                size_in=args.dim, size_out=args.dim
            )
            nengo.Connection(spatial_cleanup, model.location_ssp.input)
        else:
            raise NotImplementedError
            # spatial_cleanup = nengo.Ensemble(n_neurons=dim * neurons_per_dim, dimensions=dim, neuron_type=nengo.LIF())
            # spatial_cleanup.encoders = encoders_mixed
            # spatial_cleanup.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
            # spatial_cleanup.intercepts = mixed_intercepts
            # nengo.Connection(
            #     spatial_cleanup,
            #     location_ssp,
            #     function=clean_vectors * 1.1,
            #     eval_points=noisy_vectors,
            #     scale_eval_points=False,
            # )

        # Circular convolution
        cconv = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=dim, invert_a=False, invert_b=False)
        nengo.Connection(model.location_ssp.output, cconv.input_a)
        nengo.Connection(model.direction_ssp.output, cconv.input_b)
        nengo.Connection(cconv.output, spatial_cleanup)

        # SSP representing the map with items
        # TODO: can just have this a fixed node, it doesn't change
        # model.map_ssp = spa.State(dim)

        # direction slider, for debugging
        direction_node = nengo.Node([0])

        # Convert chosen angle into a SSP corresponding to a point on a unit circle
        nengo.Connection(direction_node, model.direction_ssp.input, function=angle_to_ssp)

        # For the location (0, 0), the vector is just a 1 in the first position, and zeros elsewhere
        nengo.Connection(initial_stim, model.location_ssp.input[0])

        nengo.Connection(model.direction_ssp.output, direction_heatmap_node)
        nengo.Connection(model.location_ssp.output, location_heatmap_node)
