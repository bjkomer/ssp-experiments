from spatial_semantic_pointers.utils import encode_point, power
import nengo_spa as spa
import numpy as np

# imports for custom SSP SPA State
from nengo.exceptions import ValidationError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import BoolParam, Default, IntParam, NumberParam

from nengo_spa.network import Network
from nengo_spa.networks import IdentityEnsembleArray
from nengo_spa.vocabulary import VocabularyOrDimParam

from ssp_navigation.utils.encodings import hilbert_2d
import nengo


# 3 directions 120 degrees apart
vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]


def to_ssp(v, X, Y):

    return encode_point(v[0], v[1], X, Y).v


def to_bound_ssp(v, item, X, Y):

    return (item * encode_point(v[0], v[1], X, Y)).v


def to_hex_region_ssp(v, X, Y, spacing=4):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for i in range(3):
        ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v

    return ret


def to_band_region_ssp(v, angle, X, Y):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for dx in np.linspace(20./63., 20, 64):
        ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v

    return ret


def grid_cell_encoder_old(dim, phases=(0, 0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    origin = np.zeros(dim, dtype='Complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi(phi=phases, angle=0, multi_phi=True)

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # origin[1+toroid_index*3+i] = phases[i]
        origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='Complex64')
    pole[:] = -1
    pole[0] = 1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # pole[1+toroid_index*3+i] = phases[i]
        pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    return (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.

    # # midpoint of the origin and the pole
    # # encoder = (origin + pole) / 2.
    # encoder = ((origin + pole) / 2.) #+ np.ones((dim,)) * 1./dim
    # # encoder = origin
    # # print('origin and pole')
    # # print(origin)
    # # print(pole)
    # #
    # # print('encoder')
    # # print(encoder)
    #
    # # encoder = np.zeros((dim, ))
    # # for n in range(n_toroids):
    # #     if n != toroid_index:
    # #         encoder +=
    #
    # return np.fft.ifft(encoder).real

def grid_cell_encoder_other_old(dim, phases=(0, 0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    encoder = np.zeros((dim,))

    n_offsets = 40
    offsets = np.linspace(0, 1, n_offsets+1)[:-1]

    for offset in offsets:

        origin = np.zeros(dim, dtype='Complex64')
        origin[:] = np.exp(1.j*2*np.pi*offset)
        origin[0] = 1

        phi_xs, phi_ys = get_sub_phi(phi=phases, angle=0, multi_phi=True)

        # modify the toroid of interest to point to the correct location
        for i in range(3):
            # origin[1+toroid_index*3+i] = phases[i]
            origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

        # set all conjugates
        origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

        pole = np.zeros(dim, dtype='Complex64')
        pole[:] = np.exp(1.j * (2 * np.pi * offset + np.pi))
        pole[0] = 1

        # fix the nyquist frequency if required
        if dim % 2 == 0:
            pole[dim // 2] = 1

        # modify the toroid of interest to point to the correct location
        for i in range(3):
            # pole[1+toroid_index*3+i] = phases[i]
            pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

        # set all conjugates
        pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

        # midpoint of the origin and the pole
        encoder += (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.

    encoder /= n_offsets

    return encoder


def grid_cell_encoder(dim, phi, angle, location=(0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    origin = np.zeros(dim, dtype='Complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi_for_loc(location=location, phi=phi, angle=angle)

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # origin[1+toroid_index*3+i] = phases[i]
        origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='Complex64')
    pole[:] = -1
    pole[0] = 1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # pole[1+toroid_index*3+i] = phases[i]
        pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    return (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.


def band_cell_encoder_old(dim, phase=0, toroid_index=0, band_index=0):
    # vector from the center of all other rings, to one specific ring on this set of rings

    origin = np.zeros(dim, dtype='complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi(phi=phase, angle=0)

    # modify the toroid of interest to point to the correct location
    origin[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='complex64')
    pole[:] = -1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    pole[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    encoder = (origin + pole) / 2.

    return np.fft.ifft(encoder).real


def band_cell_encoder(dim, phi, angle, location=(0, 0), toroid_index=0, band_index=0):
    # vector from the center of all other rings, to one specific ring on this set of rings

    origin = np.zeros(dim, dtype='complex64')
    origin[:] = 1

    # phi_xs, phi_ys = get_sub_phi(phi=phase, angle=0)
    phi_xs, phi_ys = get_sub_phi_for_loc(location=location, phi=phi, angle=angle)

    # modify the toroid of interest to point to the correct location
    origin[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='complex64')
    pole[:] = -1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    pole[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    encoder = (origin + pole) / 2.

    return np.fft.ifft(encoder).real


def orthogonal_hex_dir_7dim(phi=np.pi / 2., angle=0, multi_phi=False):
    if multi_phi:
        phi_x = phi[0]
        phi_y = phi[1]
        phi_z = phi[2]
    else:
        phi_x = phi
        phi_y = phi
        phi_z = phi

    dim = 7
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j * phi_x)
    xf[2] = 1
    xf[3] = 1
    xf[4] = np.conj(xf[3])
    xf[5] = np.conj(xf[2])
    xf[6] = np.conj(xf[1])

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = 1
    yf[2] = np.exp(1.j * phi_y)
    yf[3] = 1
    yf[4] = np.conj(yf[3])
    yf[5] = np.conj(yf[2])
    yf[6] = np.conj(yf[1])

    zf = np.zeros((dim,), dtype='Complex64')
    zf[0] = 1
    zf[1] = 1
    zf[2] = 1
    zf[3] = np.exp(1.j * phi_z)
    zf[4] = np.conj(zf[3])
    zf[5] = np.conj(zf[2])
    zf[6] = np.conj(zf[1])

    Xh = np.fft.ifft(xf).real
    Yh = np.fft.ifft(yf).real
    Zh = np.fft.ifft(zf).real

    # checks to make sure everything worked correctly
    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    assert np.allclose(np.fft.fft(Xh), xf)
    assert np.allclose(np.fft.fft(Yh), yf)
    assert np.allclose(np.linalg.norm(Xh), 1)
    assert np.allclose(np.linalg.norm(Yh), 1)

    axis_sps = [
        spa.SemanticPointer(data=Xh),
        spa.SemanticPointer(data=Yh),
        spa.SemanticPointer(data=Zh),
    ]

    n = 3
    points_nd = np.eye(n) * np.sqrt(n)
    # points in 2D that will correspond to each axis, plus one at zero
    points_2d = np.zeros((n, 2))
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1] + angle
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]

    X = power(axis_sps[0], x_axis[0])
    Y = power(axis_sps[0], y_axis[0])
    for i in range(1, n):
        X *= power(axis_sps[i], x_axis[i])
        Y *= power(axis_sps[i], y_axis[i])

    sv = transform_mat[3][0]
    return X, Y, sv


def orthogonal_hex_dir(phis=(np.pi / 2., np.pi/10.), angles=(0, np.pi/3.), even_dim=False):
    n_scales = len(phis)
    dim = 6*n_scales + 1
    if even_dim:
        dim += 1

    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1

    if even_dim:
        xf[dim//2] = 1
        yf[dim//2] = 1

    for i in range(n_scales):
        phi_xs, phi_ys = get_sub_phi(phis[i], angles[i])
        xf[1 + i * 3:1 + (i + 1) * 3] = phi_xs
        yf[1 + i * 3:1 + (i + 1) * 3] = phi_ys

    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])

    X = np.fft.ifft(xf).real
    Y = np.fft.ifft(yf).real

    return spa.SemanticPointer(data=X), spa.SemanticPointer(data=Y)


def get_sub_phi(phi, angle, multi_phi=False):
    X, Y, sv = orthogonal_hex_dir_7dim(phi=phi, angle=angle, multi_phi=multi_phi)

    xf = np.fft.fft(X.v)
    yf = np.fft.fft(Y.v)

    # xf = np.fft.fft(X.v)**(1./sv)
    # yf = np.fft.fft(Y.v)**(1./sv)

    # xf = np.fft.fft(X.v)**(sv)
    # yf = np.fft.fft(Y.v)**(sv)

    return xf[1:4], yf[1:4]


def get_sub_phi_for_loc(location=(0, 0), phi=2*np.pi, angle=0):
    X, Y, sv = orthogonal_hex_dir_7dim(phi=phi, angle=angle, multi_phi=False)

    xf = np.fft.fft(X.v)**location[0]
    yf = np.fft.fft(Y.v)**location[1]

    return xf[1:4], yf[1:4]


def orthogonal_unitary(dim, index, phi):

    fv = np.zeros(dim, dtype='complex64')
    fv[:] = 1
    fv[index] = np.exp(1.j*phi)
    fv[-index] = np.conj(fv[index])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def get_coord_rot_mat(dim=8):
    """
    Coordinate rotation matrix to make each circle orthogonal and reduce dimensionality
    """
    n_indices = (dim-1)//2

    # origin of the circle
    origin_points = np.zeros((n_indices, dim))
    # vector from origin of circle to origin of arc
    cross_vectors = np.zeros((n_indices, dim))
    # vector from origin of circle to 1/4 around the arc
    angle_vectors = np.zeros((n_indices, dim))

    u_crosses = np.zeros((n_indices, dim))
    u_angles = np.zeros((n_indices, dim))

    # the starting point on the circle is the same in all cases
    arc_origin = np.zeros((dim, ))
    arc_origin[0] = 1

    for index in range(n_indices):
        u_cross = orthogonal_unitary(dim, index + 1, np.pi)
        u_angle = orthogonal_unitary(dim, index + 1, np.pi/2.)

        # also keeping track of the axis vectors themselves, to see if they are all orthogonal
        u_crosses[index, :] = u_cross
        u_angles[index, :] = u_angle

        # midpoint of opposite ends of the circle
        origin_points[index, :] = (arc_origin + u_cross) / 2.
        cross_vectors[index, :] = arc_origin - origin_points[index, :]
        angle_vectors[index, :] = u_angle - origin_points[index, :]

        # print(np.dot(cross_vectors[index, :], angle_vectors[index, :]))
        # assert np.allclose(np.dot(cross_vectors[index, :], angle_vectors[index, :]), 0)
        assert np.abs(np.dot(cross_vectors[index, :], angle_vectors[index, :])) < 0.0000001

    all_vectors = np.vstack([cross_vectors, angle_vectors])

    for i in range(n_indices*2):
        for j in range(i+1, n_indices*2):
            # print(np.dot(all_vectors[i, :], all_vectors[j, :]))
            assert np.abs(np.dot(all_vectors[i, :], all_vectors[j, :])) < 0.0000001

    rot_dim = all_vectors.shape[0]

    transform_mat = np.linalg.lstsq(np.eye(rot_dim), all_vectors)

    # print(np.round(all_vectors @ transform_mat[0].T, 2))

    return transform_mat[0]


class SSPState(Network):
    """Represents a single vector, with optional memory.
    This is a minimal SPA network, useful for passing data along (for example,
    visual input).
    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    subdimensions : int, optional (Default: 16)
        The dimension of the individual ensembles making up the vector.
        Must divide *dimensions* evenly. The number of sub-ensembles
        will be ``dimensions // subdimensions``.
    neurons_per_dimension : int, optional (Default: 50)
        Number of neurons per dimension. Each ensemble will have
        ``neurons_per_dimension * subdimensions`` neurons, for a total of
        ``neurons_per_dimension * dimensions`` neurons.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    represent_cc_identity : bool, optional
        Whether to use optimizations to better represent the circular
        convolution identity vector. If activated, the `.IdentityEnsembleArray`
        will be used internally, otherwise a normal
        `nengo.networks.EnsembleArray` split up regularly according to
        *subdimensions*.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    **kwargs : dict
        Keyword arguments passed through to `nengo_spa.Network`.
    Attributes
    ----------
    input : nengo.Node
        Input.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam("vocab", default=None, readonly=True)
    subdimensions = IntParam("subdimensions", default=16, low=1, readonly=True)
    neurons_per_dimension = IntParam(
        "neurons_per_dimension", default=50, low=1, readonly=True
    )
    feedback = NumberParam("feedback", default=0.0, readonly=True)
    feedback_synapse = NumberParam("feedback_synapse", default=0.1, readonly=True)
    represent_cc_identity = BoolParam(
        "represent_cc_identity", default=True, readonly=True
    )

    def __init__(
        self,
        phis,
        angles,
        encoder_rng=np.random,
        vocab=Default,
        # subdimensions=Default,
        neurons_per_dimension=Default,
        feedback=Default,
        represent_cc_identity=Default,
        feedback_synapse=Default,
        limit_low=-5,
        limit_high=-5,
        **kwargs
    ):
        super(SSPState, self).__init__(**kwargs)

        self.vocab = vocab
        self.subdimensions = 6
        self.neurons_per_dimension = neurons_per_dimension
        self.feedback = feedback
        self.feedback_synapse = feedback_synapse
        self.represent_cc_identity = represent_cc_identity

        dimensions = self.vocab.dimensions

        coord_rot_mat = get_coord_rot_mat(dimensions)
        inv_coord_rot_mat = np.linalg.pinv(coord_rot_mat)

        origin = np.zeros((dimensions,))
        origin[0] = 1
        rot_origin = origin @ coord_rot_mat.T
        origin_back = rot_origin @ inv_coord_rot_mat.T
        offset_vec = origin - origin_back

        # this offset only works for odd dimensions
        # offset_vec = np.ones((dimensions,)) * 1. / dimensions

        if ((dimensions - 1) % self.subdimensions != 0) and ((dimensions - 2) % self.subdimensions != 0):
            raise ValidationError(
                "Dimensions (%d) must be divisible by subdimensions (%d)"
                % (dimensions, self.subdimensions),
                attr="dimensions",
                obj=self,
            )

        with self:
            # if self.represent_cc_identity:
            #     self.state_ensembles = IdentityEnsembleArray(
            #         self.neurons_per_dimension,
            #         dimensions,
            #         self.subdimensions,
            #         label="ssp state",
            #     )
            # else:
            #     self.state_ensembles = EnsembleArray(
            #         self.neurons_per_dimension * self.subdimensions,
            #         dimensions // self.subdimensions,
            #         ens_dimensions=self.subdimensions,
            #         eval_points=nengo.dists.CosineSimilarity(dimensions + 2),
            #         intercepts=nengo.dists.CosineSimilarity(dimensions + 2),
            #         label="ssp state",
            #     )

            # the dimensionality with the constant(s) removed
            reduced_dim = coord_rot_mat.shape[0]
            n_toroids = len(phis)
            assert n_toroids == reduced_dim // self.subdimensions

            self.state_ensembles = EnsembleArray(
                self.neurons_per_dimension * self.subdimensions,
                n_toroids,
                ens_dimensions=self.subdimensions,
                radius=2/dimensions,
                # eval_points=nengo.dists.CosineSimilarity(dimensions + 2),
                # intercepts=nengo.dists.CosineSimilarity(dimensions + 2),
                label="ssp state",
            )
            n_neurons = self.neurons_per_dimension * self.subdimensions
            # set the intercepts/encoders/eval points based on orientation and angle
            for k in range(n_toroids):
                preferred_locations = hilbert_2d(limit_low, limit_high, n_neurons, encoder_rng, p=8, N=2, normal_std=3)
                encoders_grid_cell = np.zeros((n_neurons, dimensions))
                for n in range(n_neurons):
                    encoders_grid_cell[n, :] = grid_cell_encoder(
                        location=preferred_locations[n, :],
                        dim=dimensions, phi=phis[k], angle=angles[k],
                        toroid_index=k
                    )

                # rotate, shift, and slice out relevant dimensions
                encoders_transformed = (encoders_grid_cell @ coord_rot_mat.T)[:, k*6:(k+1)*6].copy()

                # self.state_ensembles.ea_ensembles[k].intercepts = nengo.dists.Uniform(0, 1)
                # self.state_ensembles.ea_ensembles[k].encoders = encoders_transformed
                # self.state_ensembles.ea_ensembles[k].eval_points = encoders_transformed

            if self.feedback is not None and self.feedback != 0.0:
                nengo.Connection(
                    self.state_ensembles.output,
                    self.state_ensembles.input,
                    transform=self.feedback,
                    synapse=self.feedback_synapse,
                )

        # Apply coordinate transform on the input and output

        self.input = nengo.Node(size_in=dimensions, label="input")
        self.output = nengo.Node(size_in=dimensions, label="output")

        # fixed offset to push the result back into the unitary space
        self.offset = nengo.Node(offset_vec)

        nengo.Connection(self.input, self.state_ensembles.input, transform=coord_rot_mat)
        nengo.Connection(self.state_ensembles.output, self.output, transform=inv_coord_rot_mat)
        nengo.Connection(self.offset, self.output)

        # self.input = self.state_ensembles.input
        # self.output = self.state_ensembles.output
        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
