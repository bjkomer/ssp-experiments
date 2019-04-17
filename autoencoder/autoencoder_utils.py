import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from spatial_semantic_pointers.utils import encode_point


# TODO: have option to choose activation function
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.input_dim)

    def forward_activations(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction, features

    def forward(self, inputs):

        return self.forward_activations(inputs)[0]


# TODO: have option for tiled linspace data rather than random
class SSPDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, limit, x_axis_sp, y_axis_sp, rng):
        self.ssps = np.zeros((n_samples, x_axis_sp.v.shape[0])).astype(np.float32)

        for i in range(n_samples):
            x = rng.uniform(-limit, limit)
            y = rng.uniform(-limit, limit)
            self.ssps[i, :] = encode_point(x, y, x_axis_sp, y_axis_sp).v

    def __getitem__(self, index):

        return self.ssps[index]

    def __len__(self):
        return self.ssps.shape[0]