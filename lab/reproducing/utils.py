import numpy as np


def pc_to_loc_v(pc_activations, centers, jitter=0.01):
    """
    Approximate decoding of place cell activations.
    Rounding to the nearest place cell center. Just to get a sense of whether the output is in the right ballpark
    :param pc_activations: activations of each place cell, of shape (n_samples, n_place_cells)
    :param centers: centers of each place cell, of shape (n_place_cells, 2)
    :param jitter: noise to add to the output, so locations on top of each other can be seen
    :return: array of the 2D coordinates that the place cell activation most closely represents
    """

    n_samples = pc_activations.shape[0]

    indices = np.argmax(pc_activations, axis=1)

    return centers[indices] + np.random.normal(loc=0, scale=jitter, size=(n_samples, 2))


def hd_to_ang_v(hd_activations, centers, jitter=0.01):
    """
    Approximate decoding of head direction cell activations.
    Rounding to the nearest head_direction cell center. Taking sin and cos to plot in 2D.
    Just to get a sense of whether the output is in the right ballpark
    :param hd_activations: activations of each head direction cell, of shape (n_samples, n_hd_cells)
    :param centers: centers of each head direction cell, of shape (n_hd_cells, 1)
    :param jitter: noise to add to the output, so locations on top of each other can be seen
    :return: array of the 2D coordinates that the place cell activation most closely represents
    """

    n_samples = hd_activations.shape[0]

    indices = np.argmax(hd_activations, axis=1)

    angs = centers[indices] + np.random.normal(loc=0, scale=jitter, size=(n_samples,))

    # Note: 1D array needs a vstack and a transpose instead of an hstack for the shape to be correct
    return np.vstack([np.cos(angs), np.sin(angs)]).T
