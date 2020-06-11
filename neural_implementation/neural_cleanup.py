import nengo
import nengo_spa as spa
from nengo.solvers import LstsqNoise, LstsqL2
from ssp_navigation.utils.encodings import add_encoding_params, hilbert_2d, get_encoding_function
import numpy as np
import argparse
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir
from spatial_semantic_pointers.utils import encode_point, get_heatmap_vectors
from spatial_semantic_pointers.plots import SpatialHeatmap
import os


def generate_cleanup_dataset(
        encoding_func,
        n_samples,
        dim,
        n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(-1, 1, -1, 1),
        seed=13,
        normalize_memory=True):
    """
    TODO: fix this description
    Create a dataset of memories that contain items bound to coordinates

    :param n_samples: number of memories to create
    :param dim: dimensionality of the memories
    :param n_items: number of items in each memory
    :param item_set: optional list of possible item vectors. If not supplied they will be generated randomly
    :param allow_duplicate_items: if an item set is given, this will allow the same item to be at multiple places
    # :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    # :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param encoding_func: function for generating the encoding
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :param normalize_memory: if true, call normalize() on the memory semantic pointer after construction
    :return: memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    # Memory containing n_items of items bound to coordinates
    memory = np.zeros((n_samples, dim))

    # SP for the item of interest
    items = np.zeros((n_samples, n_items, dim))

    # Coordinate for the item of interest
    coords = np.zeros((n_samples * n_items, 2))

    # Clean ground truth SSP
    clean_ssps = np.zeros((n_samples * n_items, dim))

    # Noisy output SSP
    noisy_ssps = np.zeros((n_samples * n_items, dim))

    for i in range(n_samples):
        memory_sp = nengo.spa.SemanticPointer(data=np.zeros((dim,)))

        # If a set of items is given, choose a subset to use now
        if item_set is not None:
            items_used = np.random.choice(item_set, size=n_items, replace=allow_duplicate_items)
        else:
            items_used = None

        for j in range(n_items):

            x = np.random.uniform(low=limits[0], high=limits[1])
            y = np.random.uniform(low=limits[2], high=limits[3])

            # pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)
            pos = nengo.spa.SemanticPointer(data=encoding_func(x, y))

            if items_used is None:
                item = nengo.spa.SemanticPointer(dim)
            else:
                item = nengo.spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i * n_items + j, 0] = x
            coords[i * n_items + j, 1] = y
            clean_ssps[i * n_items + j, :] = pos.v
            memory_sp += (pos * item)

        if normalize_memory:
            memory_sp.normalize()

        memory[i, :] = memory_sp.v

        # Query for each item to get the noisy SSPs
        for j in range(n_items):
            noisy_ssps[i * n_items + j, :] = (memory_sp * ~nengo.spa.SemanticPointer(data=items[i, j, :])).v

    return clean_ssps, noisy_ssps, coords


parser = argparse.ArgumentParser()

parser.add_argument('--neurons-per-dim', type=int, default=25)
parser.add_argument('--n-samples', type=int, default=100)
# parser.add_argument('--n-items', type=int, default=10)
parser.add_argument('--n-items', type=int, default=7)
parser.add_argument('--res', type=int, default=32)

parser = add_encoding_params(parser)

args = parser.parse_args([])

args.spatial_encoding = 'sub-toroid-ssp'
args.dim = 256

n_neurons = args.dim * args.neurons_per_dim


enc_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)

X = spa.SemanticPointer(data=enc_func(1, 0))
Y = spa.SemanticPointer(data=enc_func(0, 1))

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


dim = len(X.v)

# get exact params of the encoding used by using the same seed
eps=0.001
n_toroids = ((args.dim - 1) // 2) // args.n_proj
rng_params = np.random.RandomState(seed=args.seed)
# Randomly select the angle for each toroid
phis = rng_params.uniform(0, 2 * np.pi, size=(n_toroids,))
angles = rng_params.uniform(-np.pi + eps, np.pi - eps, size=n_toroids)

rng = np.random.RandomState(seed=13)

preferred_locations = hilbert_2d(-args.limit, args.limit, n_neurons, rng, p=8, N=2, normal_std=3)

encoders_place_cell = np.zeros((n_neurons, args.dim))
encoders_band_cell = np.zeros((n_neurons, args.dim))
encoders_grid_cell = np.zeros((n_neurons, args.dim))
encoders_mixed = np.zeros((n_neurons, args.dim))
mixed_intercepts = []
for n in range(n_neurons):
    ind = rng.randint(0, len(phis))
    encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])

    encoders_grid_cell[n, :] = grid_cell_encoder(
        location=preferred_locations[n, :],
        dim=args.dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind
    )

    band_ind = rng.randint(0, 3)
    encoders_band_cell[n, :] = band_cell_encoder(
        location=preferred_locations[n, :],
        dim=args.dim, phi=phis[ind], angle=angles[ind],
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


cache_fname = 'neural_cleanup_dataset_{}.npz'.format(dim)
if os.path.exists(cache_fname):
    data = np.load(cache_fname)
    clean_vectors = data['clean_vectors']
    noisy_vectors = data['noisy_vectors']
    coords = data['coords']
    heatmap_vectors = data['heatmap_vectors']
else:
    heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
    # n_samples = 50000
    # clean_vectors = np.zeros((n_samples, dim))
    # noisy_vectors = np.zeros((n_samples, dim))
    # # locations = rng.uniform(-limit, limit, size=(n_samples, 2))
    # locations = rng.uniform(-args.limit, args.limit, size=(n_samples, 2))
    # noise = rng.normal(0, 0.5/np.sqrt(dim), size=(n_samples, dim))
    # for i in range(n_samples):
    #     clean_vectors[i, :] = encode_point(locations[i, 0], locations[i, 1], X, Y).v
    #     noisy_vectors[i, :] = clean_vectors[i, :] + noise[i, :]

    clean_vectors, noisy_vectors, coords = generate_cleanup_dataset(
        enc_func,
        args.n_samples,
        args.dim,
        args.n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(-args.limit, args.limit, -args.limit, args.limit),
        seed=13,
        normalize_memory=True
    )

    np.savez(
        cache_fname,
        clean_vectors=clean_vectors,
        noisy_vectors=noisy_vectors,
        coords=coords,
        heatmap_vectors=heatmap_vectors,
    )

model = nengo.Network(seed=13)
with model:

    ssp_input = nengo.Node(
        lambda t: noisy_vectors[int(np.floor(t*10.)), :],
        size_in=0,
        size_out=args.dim,

    )

    coord_input = nengo.Node(
        lambda t: coords[int(np.floor(t*10.)), :],
        size_in=0,
        size_out=2,
    )

    spatial_cleanup = nengo.Ensemble(
        n_neurons=args.dim * args.neurons_per_dim, dimensions=args.dim, neuron_type=nengo.LIF()
    )
    location_ssp = nengo.Ensemble(
        n_neurons=args.dim * args.neurons_per_dim, dimensions=args.dim, neuron_type=nengo.LIF()
    )

    nengo.Connection(ssp_input, spatial_cleanup)

    spatial_cleanup.encoders = encoders_mixed
    # spatial_cleanup.encoders = encoders_grid_cell
    spatial_cleanup.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
    spatial_cleanup.intercepts = mixed_intercepts

    location_ssp.encoders = encoders_mixed
    # location_ssp.encoders = encoders_grid_cell
    location_ssp.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
    location_ssp.intercepts = mixed_intercepts

    nengo.Connection(
        spatial_cleanup,
        location_ssp,
        function=clean_vectors,
        eval_points=noisy_vectors,
        scale_eval_points=False,
        # solver=LstsqL2(),
    )

    if __name__ == '__main__':
        p_clean = nengo.Probe(location_ssp)
        # p_coord = nengo.Probe(coord_input)
    else:
        # vmin = None
        # vmax = None
        vmin = -1
        vmax = 1
        location_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=dim, size_out=0,
        )
        nengo.Connection(location_ssp, location_heatmap_node)

        spatial_cleanup_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=dim, size_out=0,
        )
        nengo.Connection(spatial_cleanup, spatial_cleanup_heatmap_node)

        input_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=dim, size_out=0,
        )
        nengo.Connection(ssp_input, input_heatmap_node)

        coord_heatmap_node = nengo.Node(
            SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
            size_in=dim, size_out=0,
        )
        nengo.Connection(coord_input, coord_heatmap_node, function=to_ssp)


if __name__ == '__main__':
    duration = 50
    sim = nengo.Simulator(model)
    sim.run(duration)

    output_fname = 'neural_cleanup_output_{}.npz'.format(dim)

    np.savez(
        output_fname,
        clean_output=sim.data[p_clean],
    )
