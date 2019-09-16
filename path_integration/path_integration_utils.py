import numpy as np
# from ssp_navigation.utils.models import LearnedEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_semantic_pointers.utils import make_good_unitary, encode_point


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


class EncodingLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512):
        super(EncodingLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)

    def forward(self, inputs):

        encoding = F.relu(self.encoding_layer(inputs))

        return encoding


def encoding_func_from_model(path,
                             # repr_dim=512, id_size=10, hidden_size=512, n_hidden_layers=1
                             ):
    # pass
    #
    # model = LearnedEncoding(
    #     input_size=repr_dim,  # this may be the only parameter that needs to be correct
    #     maze_id_size=id_size,
    #     hidden_size=hidden_size,
    #     output_size=2,
    #     n_layers=n_hidden_layers
    # )

    encoding_layer = EncodingLayer()

    # TODO: modify this to have a network that just does the encoding
    # TODO: make sure this is working correctly
    print("Loading learned first layer parameters from pretrained model")
    state_dict = torch.load(path)

    for name, param in state_dict.items():
        if name in ['encoding_layer.weight', 'encoding_layer.bias']:
            encoding_layer.state_dict()[name].copy_(param)

    print("Freezing first layer parameters for training")
    for name, param in encoding_layer.named_parameters():
        if name in ['encoding_layer.weight', 'encoding_layer.bias']:
            param.requires_grad = False
        if param.requires_grad:
            print(name)

    # def encoding_func(x, y):
    def encoding_func(positions):
        return encoding_layer(torch.Tensor(positions)).detach().numpy()

    return encoding_func


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def pc_gauss_encoding_func(limit_low=0, limit_high=1, dim=512, sigma=0.25, use_softmax=False, rng=np.random):
    # generate PC centers
    pc_centers = rng.uniform(low=limit_low, high=limit_high, size=(dim, 2))

    # TODO: make this more efficient
    def encoding_func(positions):
        activations = np.zeros((dim,))
        for i in range(dim):
            activations[i] = np.exp(-((pc_centers[i, 0] - positions[0]) ** 2 + (pc_centers[i, 1] - positions[1]) ** 2) / sigma / sigma)
        if use_softmax:
            return softmax(activations)
        else:
            return activations

    return encoding_func


def ssp_encoding_func(seed=13, dim=512, ssp_scaling=1):
    rng = np.random.RandomState(seed=seed)

    x_axis_sp = make_good_unitary(dim=dim, rng=rng)
    y_axis_sp = make_good_unitary(dim=dim, rng=rng)

    def encoding_func(positions):
        return encode_point(
            x=positions[0]*ssp_scaling,
            y=positions[1]*ssp_scaling,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp
        ).v

    return encoding_func


def hd_gauss_encoding_func(limit_low=-np.pi, limit_high=np.pi, dim=12, sigma=0.25, use_softmax=False, rng=np.random):
    # generate PC centers
    hd_centers = rng.uniform(low=limit_low, high=limit_high, size=(dim, 1))

    # TODO: make this more efficient
    def encoding_func(angle):
        activations = np.zeros((dim,))
        for i in range(dim):
            # include shifts of 2pi and -2pi to handle the cyclic nature correctly
            activations[i] = np.exp(-((hd_centers[i] - angle) ** 2) / sigma / sigma) +\
                             np.exp(-((hd_centers[i] - angle + 2 * np.pi) ** 2) / sigma / sigma) + \
                             np.exp(-((hd_centers[i] - angle - 2 * np.pi) ** 2) / sigma / sigma)
        if use_softmax:
            return softmax(activations)  # NOTE: there may be some issues with softmax and wrapping
        else:
            return activations

    return encoding_func
