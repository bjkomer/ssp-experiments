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
