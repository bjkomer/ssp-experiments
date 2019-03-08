import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Single hidden layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # For compatibility with DeepRL code
        self.feature_dim = hidden_size #output_size #hidden_size #FIXME, this should be hidden size, the downstream code needs modification

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction, features


class MLP(nn.Module):
    """
    Multi-layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512, n_layers=2):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.inner_layers = []

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        for i in range(self.n_layers - 1):
            features = F.relu(self.inner_layers[i](features))
        prediction = self.output_layer(features)

        return prediction
