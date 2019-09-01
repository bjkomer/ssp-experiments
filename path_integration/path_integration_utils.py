import numpy as np
# from ssp_navigation.utils.models import LearnedEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F


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
