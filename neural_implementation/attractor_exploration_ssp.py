import nengo
import nengo_spa as spa
import numpy as np
from nengo.processes import Piecewise
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors
from encoders import grid_cell_encoder, band_cell_encoder
from ssp_navigation.utils.encodings import hilbert_2d


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


dim = 5
phis_x = np.array([np.pi/2., 0])
phis_y = np.array([0, np.pi/2.])

X = unitary_by_phi(phis_x)
Y = unitary_by_phi(phis_y)
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

n_neurons = 2500
rng = np.random.RandomState(seed=13)


def to_ssp(v):

    return encode_point(v[0], v[1], X, Y).v


def to_ssp_with_offset(v):

    return encode_point(v[0], v[1], X, Y).v - offset


def input_ssp(v):

    if v[2] > .5:
        return encode_point(v[0], v[1], X, Y).v - offset
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
# encoders_band_cell = np.zeros((n_neurons, dim))
# encoders_grid_cell = np.zeros((n_neurons, dim))
for n in range(n_neurons):
    # encoders_place_cell[n, :] = to_ssp(preferred_locations[n, :])
    encoders_place_cell[n, :] = to_ssp_with_offset(preferred_locations[n, :])
    # encoders_band_cell[n, :] = band_region_ssp(preferred_locations[n, :], angle=rng.uniform(0, 2*np.pi))
    # spacing = rng.choice(spacings)
    # angle = rng.uniform(0, 2*np.pi)
    # encoders_grid_cell[n, :] = to_hex_region_ssp(preferred_locations[n, :], spacing=spacing, angle=angle)


def feedback_real(x):
    # calculate current phase for derivative
    xf = np.fft.fft(x + offset)
    phase = np.log(xf).imag
    deriv = dv_dxy(dim, phis_x, phis_y, phase=phase[1:])

    # x_vel = 1*10
    # y_vel = 1.5*50
    x_vel = 1
    y_vel = 1.5
    return x * 1.1
    # return x*1.10 + tau * x_vel * deriv[:, 0] + tau * y_vel * deriv[:, 1]
    

model = nengo.Network(seed=15)
with model:

    kick = nengo.Node([0]*dim)

    # # in complex fourier coefficient space
    # ssp = nengo.Ensemble(n_neurons=1000, dimensions=4)
    # nengo.Connection(ssp, ssp, function=feedback_fourier, synapse=tau)

    # in real space
    ssp = nengo.Ensemble(n_neurons=n_neurons, dimensions=5)
    ssp.encoders = encoders_place_cell
    ssp.eval_points = encoders_place_cell
    # ssp.intercepts = [0.25]*n_neurons
    ssp.intercepts = [0.50] * n_neurons
    feedback_conn = nengo.Connection(
        ssp,
        ssp,
        function=feedback_real,
        synapse=tau,
        solver=nengo.solvers.LstsqL2(weights=True),
    )

    nengo.Connection(kick, ssp)

    heatmap_node = nengo.Node(
        SpatialHeatmap(heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None),
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

    if weights.shape[0] == 5:
        weights = encoders_place_cell @ weights

    plt.figure()
    plt.imshow(weights)

    sqrt_neurons = int(np.sqrt(n_neurons))
    plt.figure()
    plt.imshow(weights[3, :].reshape(sqrt_neurons, sqrt_neurons))

    plt.show()


