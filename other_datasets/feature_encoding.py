import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, power


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
