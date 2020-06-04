from encoders import orthogonal_hex_dir, SSPState
import nengo_spa as spa
import nengo
import numpy as np
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.utils import encode_point, get_heatmap_vectors
import os

dim = 7#272
dim = 6*12+1

T = (dim+1)//2
n_toroid = T // 3
axis_rng = np.random.RandomState(seed=14)
phis = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)
angles = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)
# phis = np.array([np.pi/2., np.pi/3., np.pi/5.])
# angles = np.array([0, 1, 2])
X, Y = orthogonal_hex_dir(phis=phis, angles=angles, even_dim=dim%2==0)

dim = len(X.v)


limit = 5
res = 256
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

hmv = get_heatmap_vectors(xs, xs, X, Y)

# if not os.path.exists('hmv_exp_{}.npz'.format(dim)):
#     # hmv = get_heatmap_vectors_hex(xs, xs, X, Y, Z)
#     hmv = get_heatmap_vectors(xs, xs, X, Y)
#     np.savez('hmv_exp_{}.npz'.format(dim), hmv=hmv)
# else:
#     hmv = np.load('hmv_exp_{}.npz'.format(dim))['hmv']


def input_ssp(v):

    if v[2] > .5:
        return encode_point(v[0], v[1], X, Y).v
        # use_offset = False
        # if use_offset:
        #     return encode_point(v[0], v[1], X, Y).v - offset
        # else:
        #     return encode_point(v[0], v[1], X, Y).v
    else:
        return np.zeros((dim,))


model = nengo.Network(seed=13)
with model:
    model.current_loc = SSPState(
        vocab=dim,
        phis=phis,
        angles=angles,
        feedback=1.01,
        limit_low=-limit,
        limit_high=limit,
    )

    # x, y, and on/off
    ssp_input = nengo.Node([0, 0, 0])

    nengo.Connection(ssp_input, model.current_loc.input, function=input_ssp)

    heatmap = SpatialHeatmap(heatmap_vectors=hmv, xs=xs, ys=xs, cmap='plasma', vmin=-1, vmax=1)
    heatmap_node = nengo.Node(
        heatmap,
        size_in=dim,
        size_out=0
    )
    nengo.Connection(model.current_loc.output, heatmap_node)

    heatmap_true = SpatialHeatmap(heatmap_vectors=hmv, xs=xs, ys=xs, cmap='plasma', vmin=None, vmax=None)
    heatmap_true_node = nengo.Node(
        heatmap_true,
        size_in=dim,
        size_out=0
    )
    nengo.Connection(ssp_input, heatmap_true_node, function=input_ssp)
