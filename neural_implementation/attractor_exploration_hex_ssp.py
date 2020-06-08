import nengo
import nengo_spa as spa
import numpy as np
from nengo.processes import Piecewise
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors, get_fixed_dim_sub_toriod_axes
from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir
from ssp_navigation.utils.encodings import hilbert_2d


use_offset = False
# use_offset = True

# vmin = 0
# vmax = 1
vmin = None
vmax = None


def dv_dxy(dim, phis_x, phis_y, phase):
    ret = np.zeros((dim, 2))
    T = (dim - 1) //2
    for n in range(dim):
        for k in range(T):
            ret[n, 0] += (-2. / dim) * phis_x[k] * np.sin(2 * np.pi * k * n / dim + phase[k])
            ret[n, 1] += (-2. / dim) * phis_y[k] * np.sin(2 * np.pi * k * n / dim + phase[k])
    return ret


# def feedback(x):
#     x0, x1, w = x  # These are the three variables stored in the ensemble
#     return x0 + w * w_max * tau * x1, x1 - w * w_max * tau * x0, 0

tau = .1
phi_0 = 1
phi_1 = 1.5


def feedback_fourier(x):
    phi_0r, phi_0i, phi_1r, phi_1i = x

    ret = np.zeros((4,))
    ret[0] = phi_0r + phi_0 * tau * phi_0i
    ret[1] = phi_0i - phi_0 * tau * phi_0r
    ret[2] = phi_1r + phi_1 * tau * phi_1i
    ret[3] = phi_1i - phi_1 * tau * phi_1r
    return ret


def unitary_by_phi(phis):
    vec_dim = len(phis)*2+1
    vf = np.ones((vec_dim, ), dtype='complex64')
    vf[0] = 1
    vf[1:(dim + 1) // 2] = np.exp(1.j*phis)
    vf[-1:dim // 2:-1] = np.conj(vf[1:(dim + 1) // 2])

    assert np.allclose(np.abs(vf), 1)
    v = np.fft.ifft(vf).real
    assert np.allclose(np.fft.fft(v), vf)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v)


dim = 13#7
# dim = 7
# dim = 31
dim = 19
dim = 43
dim = 25
# dim = 13
# phis_x = np.array([np.pi/2., 0])
# phis_y = np.array([0, np.pi/2.])
#
# X = unitary_by_phi(phis_x)
# Y = unitary_by_phi(phis_y)

axis_rng = np.random.RandomState(seed=14)
# scale_ratio = 0 #2
# start_index = 0
# X, Y = get_fixed_dim_sub_toriod_axes(dim=dim, scale_ratio=scale_ratio, scale_start_index=start_index, rng=axis_rng)

# phis = (np.pi*.75, np.pi / 2., np.pi/3., np.pi/5., np.pi*.4, np.pi*.6, np.pi*.15)
# angles = (0, np.pi*.3, np.pi*.2, np.pi*.4, np.pi*.1, np.pi*.5, np.pi*.7)

T = (dim+1)//2
n_toroid = T // 3
# phis = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)
# angles = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)

if dim == 7:
    phis = (np.pi*.5,)
    angles = (0,)
elif dim == 13:
    phis = (np.pi*.5, np.pi*.3)
    angles = (0, np.pi/6.)
elif dim == 19:
    phis = (np.pi*.5, np.pi*.3, np.pi*.75)
    angles = (0, np.pi/6., np.pi/12.)
else:
    phis = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)
    angles = axis_rng.uniform(-np.pi, np.pi, size=n_toroid)

X, Y = orthogonal_hex_dir(phis=phis, angles=angles)

print(len(X.v))

# retrieve phis from the axis vectors
xf = np.fft.fft(X.v)
yf = np.fft.fft(Y.v)

phis_x = np.log(xf[1:T+1]).imag
phis_y = np.log(yf[1:T+1]).imag

limit = 5
res = 64
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

offset = np.ones((dim,))*1./dim


def get_heatmap_vectors_with_offset(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            )
            vectors[i, j, :] = p.v - offset

    return vectors



# heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
heatmap_vectors = get_heatmap_vectors_with_offset(xs, ys, X, Y)

neurons_per_dim = 100#200
n_neurons = neurons_per_dim * dim
# n_neurons = 2500
rng = np.random.RandomState(seed=13)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


def to_ssp_with_offset(v):

    return encode_point(v[0], v[1], X, Y).v - offset


def input_ssp(v):

    if v[2] > .5:
        if use_offset:
            return encode_point(v[0], v[1], X, Y).v - offset
        else:
            return encode_point(v[0], v[1], X, Y).v
    else:
        return np.zeros((dim,))

tiled_neurons = True

if tiled_neurons:
    n_neurons = int(np.sqrt(n_neurons))**2
    sqrt_neurons = int(np.sqrt(n_neurons))
    preferred_locations = np.zeros((n_neurons, 2))
    xsn = np.linspace(-limit, limit, sqrt_neurons)
    ysn = np.linspace(-limit, limit, sqrt_neurons)
    for i, x in enumerate(xsn):
        for j, y in enumerate(ysn):
            preferred_locations[i*sqrt_neurons + j, :] = np.array([x, y])
else:
    preferred_locations = hilbert_2d(-limit, limit, n_neurons, rng, p=8, N=2, normal_std=3)

encoders_place_cell = np.zeros((n_neurons, dim))
encoders_band_cell = np.zeros((n_neurons, dim))
encoders_grid_cell = np.zeros((n_neurons, dim))
encoders_mixed = np.zeros((n_neurons, dim))
grid_inds = np.zeros((n_neurons, ))
mixed_intercepts = []
for n in range(n_neurons):
    # encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])
    if use_offset:
        encoders_place_cell[n, :] = to_ssp_with_offset(preferred_locations[n, :])
    else:
        encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])

    ind = rng.randint(0, len(phis))
    grid_inds[n] = ind
    encoders_grid_cell[n, :] = grid_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind
    )

    if use_offset:
        encoders_grid_cell[n, :] -= offset

    band_ind = rng.randint(0, 3)
    encoders_band_cell[n, :] = band_cell_encoder(
        location=preferred_locations[n, :],
        dim=dim, phi=phis[ind], angle=angles[ind],
        toroid_index=ind,
        band_index=band_ind
    )

    if use_offset:
        encoders_band_cell[n, :] -= offset

    mix_ind = rng.randint(0, 3)
    if mix_ind == 0:
        encoders_mixed[n, :] = encoders_place_cell[n, :]
        # mixed_intercepts.append(.4)
        mixed_intercepts.append(.75)
    elif mix_ind == 1:
        encoders_mixed[n, :] = encoders_grid_cell[n, :]
        # mixed_intercepts.append(.2)
        mixed_intercepts.append(.5)
    elif mix_ind == 2:
        encoders_mixed[n, :] = encoders_band_cell[n, :]
        # mixed_intercepts.append(0.)
        mixed_intercepts.append(.25)

    # metadata[n, 0] = ind
    # metadata[n, 1] = band_ind
    # metadata[n, 2] = mix_ind

# place cell evaluation points
n_eval_points = 5000
eval_point_locations = rng.uniform(-limit, limit, size=(n_eval_points, 2))
eval_points = np.zeros((n_eval_points, dim))
eval_noise = rng.normal(0, 0.1/np.sqrt(dim), size=(n_eval_points, dim))
for i in range(n_eval_points):
    if use_offset:
        eval_points[i, :] = to_ssp_with_offset(eval_point_locations[i, :])
    else:
        eval_points[i, :] = to_ssp(eval_point_locations[i, :])

# add some noise to the points
eval_points = eval_points + eval_noise


def feedback_real(x):
    # calculate current phase for derivative
    if use_offset:
        xf = np.fft.fft(x + offset)
    else:
        xf = np.fft.fft(x)
    phase = np.log(xf).imag
    deriv = dv_dxy(dim, phis_x, phis_y, phase=phase[1:])

    # x_vel = 1*10
    # y_vel = 1.5*50
    x_vel = 1
    y_vel = 1.5
    # return x * 1.1
    return x * 1.01
    # return x*1.10 + tau * x_vel * deriv[:, 0] + tau * y_vel * deriv[:, 1]
    

model = nengo.Network(seed=15)
with model:

    kick = nengo.Node([0]*dim)

    # # in complex fourier coefficient space
    # ssp = nengo.Ensemble(n_neurons=1000, dimensions=4)
    # nengo.Connection(ssp, ssp, function=feedback_fourier, synapse=tau)

    # in real space
    ssp = nengo.Ensemble(n_neurons=n_neurons, dimensions=dim)
    # ssp.encoders = encoders_place_cell
    # ssp.encoders = encoders_mixed
    ssp.encoders = encoders_grid_cell
    # ssp.eval_points = encoders_place_cell
    # ssp.eval_points = np.vstack([encoders_place_cell, encoders_band_cell, encoders_grid_cell])
    # ssp.eval_points = encoders_place_cell
    ssp.eval_points = eval_points
    # ssp.intercepts = [0.25]*n_neurons
    # ssp.intercepts = [0.0] * n_neurons
    ssp.intercepts = rng.uniform(0, .75, size=(n_neurons,))
    # ssp.intercepts = mixed_intercepts
    # ssp.intercepts = [0.50] * n_neurons
    feedback_conn = nengo.Connection(
        ssp,
        ssp,
        function=feedback_real,
        synapse=tau,
        solver=nengo.solvers.LstsqL2(weights=True),
    )

    nengo.Connection(kick, ssp)

    heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=vmin, vmax=vmax),
        size_in=dim, size_out=0,
    )

    nengo.Connection(ssp, heatmap_node)

    # x, y, and on/off
    ssp_input = nengo.Node([0, 0, 0])

    nengo.Connection(ssp_input, ssp, function=input_ssp)

    p_weights = nengo.Probe(feedback_conn, 'weights')

if __name__ == '__main__':
    sim = nengo.Simulator(model)
    sim.run(0.01)
    weights = sim.data[p_weights][-1]
    print(weights.shape)
    import matplotlib.pyplot as plt

    if weights.shape[0] == dim:
        weights = ssp.encoders @ weights

    plt.figure()
    plt.imshow(weights)
    plt.figure()
    weights_grid0 = weights[0, :].copy()
    weights_grid0[grid_inds==0] = 0
    weights_grid1 = weights[0, :].copy()
    weights_grid1[grid_inds==1] = 0
    sqrt_neurons = int(np.sqrt(n_neurons))
    plt.imshow(weights[0, :].reshape(sqrt_neurons, sqrt_neurons))

    plt.figure()
    plt.imshow(weights_grid0.reshape(sqrt_neurons, sqrt_neurons))

    plt.figure()
    plt.imshow(weights_grid1.reshape(sqrt_neurons, sqrt_neurons))



    plt.show()
