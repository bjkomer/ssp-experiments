import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, power
from scipy.special import legendre


# Note: this is extremely slow, may want to 'cache' the result on real experiments
def encode_dataset(data, dim=256, seed=13, scale=1.0):
    """
    :param data: the data to be encoded
    :param dim: dimensionality of the SSP
    :param seed: seed for the single axis vector
    :param scale: scaling of the data for the encoding
    :return:
    """
    rng = np.random.RandomState(seed=seed)
    # TODO: have option to normalize everything first, for consistent relative scale
    axis_vec = make_good_unitary(dim, rng=rng)

    n_samples = data.shape[0]
    n_features = data.shape[1]

    n_out_features = n_features * dim

    data_out = np.zeros((n_samples, n_out_features))

    for s in range(n_samples):
        for f in range(n_features):
            data_out[s, f*dim:(f+1)*dim] = power(axis_vec, data[s, f] * scale).v

    return data_out


# Note: this is extremely slow, may want to 'cache' the result on real experiments
def encode_dataset_nd(data, dim=256, seed=13, scale=1.0, style='normal'):
    """
    :param data: the data to be encoded
    :param dim: dimensionality of the SSP
    :param seed: seed for the single axis vector
    :param scale: scaling of the data for the encoding
    :param style: 'normal' or 'simplex'
    :return:
    """

    n_samples = data.shape[0]
    n = data.shape[1]

    data_out = np.zeros((n_samples, dim))

    if style == 'normal':
        encoding_func = get_nd_encoding_func(n=n, dim=dim, seed=seed)
    elif style == 'simplex':
        encoding_func = get_nd_simplex_encoding_func(n=n, dim=dim, seed=seed)
    else:
        raise NotImplementedError

    for s in range(n_samples):
        data_out[s, :] = encoding_func(data[s, :] * scale)

    return data_out


def get_simplex_coordinates(n):
    # https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_regular_n-dimensional_simplex_in_Rn

    # n+1 vectors in n dimensions define the vertices of the shape
    axes = np.zeros((n + 1, n))

    # the dot product between any two vectors must be this value
    dot_product = -1/n

    # initialize the first vector to [1, 0, 0 ...]
    axes[0, 0] = 1
    axes[1:, 0] = dot_product
    axes[0, 1:] = 0

    # element index
    for ei in range(1, n):
        # print(axes[ei, :ei])
        # calculated using pythagorean theorem, distance to center must be 1

        prev_sum = np.sum(axes[ei, :ei]**2)

        axes[ei, ei] = np.sqrt(1 - prev_sum)  # 1**2 = ?**2 + prev**2 + .. -> ? = sqrt(1 - prev**2 ...)

        # set this element in other vectors based on the dot product
        axes[ei+1:, ei] = (dot_product - prev_sum) / axes[ei, ei]  # dp = new*? + prev**2 + ... -> ? = (dp - prev**2 + ...) / new

        # set all other elements in the vector to 0
        axes[ei, ei + 1:] = 0

    # the last vector is the second last vector, but with the sign flipped on the last element
    axes[-1, :] = axes[-2, :]
    axes[-1, -1] = -axes[-1, -1]

    return axes


def get_nd_simplex_encoding_func(n, dim, seed=13):

    transform_axes = get_simplex_coordinates(n)

    rng = np.random.RandomState(seed=seed)

    axis_vectors = []
    for i in range(n + 1):
        axis_vectors.append(make_good_unitary(dim, rng=rng))

    def encoding_func(features):
        """
        Take in 'n' features as a numpy array, and output a 'dim' dimensional SSP
        """
        # TODO: any scaling required?
        # TODO: make sure matrix multiply order is correct
        tranformed_features = transform_axes @ features

        vec = power(axis_vectors[0], tranformed_features[0])
        for i in range(1, n + 1):
            vec *= power(axis_vectors[i], tranformed_features[i])

        return vec.v

    return encoding_func


def get_nd_encoding_func(n, dim, seed=13):

    rng = np.random.RandomState(seed=seed)

    axis_vectors = []
    for i in range(n):
        axis_vectors.append(make_good_unitary(dim, rng=rng))

    def encoding_func(features):
        """
        Take in 'n' features as a numpy array, and output a 'dim' dimensional SSP
        """

        vec = power(axis_vectors[0], features[0])
        for i in range(1, n):
            vec *= power(axis_vectors[i], features[i])

        return vec.v

    return encoding_func


def get_one_hot_encoding_func(dim, limit_low=-1, limit_high=1, **_):

    xs = np.linspace(limit_low, limit_high, dim)

    def encoding_func(feature):
        arr = np.zeros((len(xs),))

        ind = (np.abs(xs - feature)).argmin()
        arr[ind] = 1

        return arr

    return encoding_func


def get_tile_coding_encoding_func(dim, n_tiles, seed, limit_low=-1, limit_high=1, **_):

    rng = np.random.RandomState(seed=seed)

    n_bins = dim // n_tiles

    assert dim == n_tiles * n_bins

    # Make the random offsets relative to the bin_size
    bin_size = (limit_high - limit_low) / (n_bins)

    # A series of shifted linspaces
    xss = np.zeros((n_tiles, n_bins))

    # The x and y offsets for each tile. Max offset is half the bin_size
    offsets = rng.uniform(-bin_size/2, bin_size/2, size=(n_tiles, 2))

    for i in range(n_tiles):
        xss[i, :] = np.linspace(limit_low + offsets[i, 0], limit_high + offsets[i, 0], n_bins)

    def encoding_func(feature):
        arr = np.zeros((n_tiles, n_bins))
        for i in range(n_tiles):
            ind = (np.abs(xss[i, :] - feature)).argmin()
            arr[i, ind] = 1
        return arr.flatten()

    return encoding_func


def get_rbf_encoding_func(dim, sigma, seed, random_centers=True, limit_low=-1, limit_high=1, **_):

    rng = np.random.RandomState(seed=seed)

    # generate PC centers
    if random_centers:
        pc_centers = rng.uniform(low=limit_low, high=limit_high, size=(dim,))
    else:
        pc_centers = np.linspace(limit_low, limit_high, dim)

    def encoding_func(feature):
        activations = np.zeros((dim,))
        for i in range(dim):
            activations[i] = np.exp(-((pc_centers[i] - feature) ** 2) / sigma / sigma)
        return activations

    return encoding_func


def get_ssp_encoding_func(dim, scale, seed, **_):

    rng = np.random.RandomState(seed=seed)

    axis_vec = make_good_unitary(dim, rng=rng)

    def encoding_func(feature):
        return power(axis_vec, feature * scale).v

    return encoding_func


def get_legendre_encoding_func(dim, limit_low=-1, limit_high=1, **_):
    """
    Encoding a ND point by expanding the dimensionality through the legendre polynomials
    (starting with order 1, ignoring the constant)
    """

    # set up legendre polynomial functions
    poly = []
    for i in range(dim):
        poly.append(legendre(i + 1))

    domain = limit_high - limit_low

    def encoding_func(feature):

        # shift the feature to be between -1 and 1 before going through the polynomials
        fn = ((feature - limit_low) / domain) * 2 - 1
        ret = np.zeros((dim,))
        for i in range(dim):
            ret[i] = poly[i](fn)

        return ret

    return encoding_func

def encode_comparison_dataset(data, encoding, dim, **params):

    if encoding == 'one-hot':
        enc_func = get_one_hot_encoding_func(dim=dim, **params)
    elif encoding == 'tile-code':
        enc_func = get_tile_coding_encoding_func(dim=dim, **params)
    elif encoding == 'pc-gauss':
        enc_func = get_rbf_encoding_func(dim=dim, random_centers=True, **params)
    elif encoding == 'pc-gauss-tiled':
        enc_func = get_rbf_encoding_func(dim=dim, random_centers=False, **params)
    elif encoding == 'legendre':
        enc_func = get_legendre_encoding_func(dim=dim, **params)
    elif encoding == 'ssp':
        enc_func = get_ssp_encoding_func(dim=dim, **params)
    else:
        raise NotImplementedError('unknown encoding: {}'.format(encoding))

    n_samples = data.shape[0]
    n_features = data.shape[1]

    n_out_features = n_features * dim

    data_out = np.zeros((n_samples, n_out_features))

    for s in range(n_samples):
        for f in range(n_features):
            data_out[s, f*dim:(f+1)*dim] = enc_func(data[s, f])

    return data_out
