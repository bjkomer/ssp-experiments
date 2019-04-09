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


class LearnedEncoding(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, maze_id_size=512, hidden_size=512, output_size=2):
        super(LearnedEncoding, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.maze_id_size = maze_id_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
        # self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
        self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        maze_id = inputs[:, :self.maze_id_size]
        loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
        goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]

        loc_encoding = F.relu(self.encoding_layer(loc_pos))
        goal_encoding = F.relu(self.encoding_layer(goal_pos))
        features = F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1)))
        prediction = self.output_layer(features)

        return prediction, features, loc_encoding, goal_encoding

    def forward(self, inputs):

        return self.forward_activations_encoding(inputs)[0]


class TwoLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, hidden_size=512, output_size=2):
        super(TwoLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
        self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):
        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations and encoding layer activations, as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features, encoding
