import argparse
import numpy as np
from arguments import add_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from trajectory_dataset import train_test_loaders

parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

torch.manual_seed(args.seed)

data = np.load('data/path_integration_trajectories_200t_15s.npz')

n_samples = 1000
rollout_length = 100
batch_size = 10
n_epochs = 5


class PathIntegrationModel(nn.Module):

    def __init__(self, input_size=3, lstm_hidden_size=128, linear_hidden_size=512,
                 unroll_length=100, n_pc=256, n_hd=12):

        super(PathIntegrationModel, self).__init__()

        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.unroll_length = unroll_length

        self.n_pc = n_pc  # number of place cells
        self.n_hd = n_hd  # number of head direction cells

        # self.lstm = nn.LSTM(
        self.lstm = nn.LSTMCell(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            # num_layers=1
        )

        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=self.linear_hidden_size,
        )

        self.dropout = nn.Dropout(p=.5)

        self.pc_output = nn.Linear(
            in_features=self.linear_hidden_size,
            out_features=self.n_pc
        )

        self.hd_output = nn.Linear(
            in_features=self.linear_hidden_size,
            out_features=self.n_hd
        )

        # Linear transforms for ground truth pc and hd activations into initial hidden and cell state of lstm
        self.w_cp = nn.Linear(
            in_features=self.n_pc,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_cd = nn.Linear(
            in_features=self.n_hd,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_hp = nn.Linear(
            in_features=self.n_pc,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_hd = nn.Linear(
            in_features=self.n_hd,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

    def forward(self, velocity_inputs, initial_pc_activation, initial_hd_activation):

        print("velocity_inputs[0].shape", velocity_inputs[0].shape)
        print("initial_pc_activation.shape", initial_pc_activation.shape)
        print("initial_hd_activation.shape", initial_hd_activation.shape)
        print("")

        # Compute initial hidden state
        cell_state = self.w_cp(initial_pc_activation) + self.w_cd(initial_hd_activation)
        hidden_state = self.w_hp(initial_pc_activation) + self.w_hd(initial_hd_activation)

        print("cell_state.shape", cell_state.shape)
        print("hidden_state.shape", hidden_state.shape)
        print("")

        # for i in range(self.unroll_length):
        # NOTE: there is a way to do this all at once without a loop:
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # for i in range(self.unroll_length):
        #     print("velocity_inputs[:, i, :].shape", velocity_inputs[:, i, :].shape)
        #     out, (hidden_state, cell_state) = self.lstm(velocity_inputs[:, i, :].view(1, 1, -1), (hidden_state, cell_state))

        for velocity in velocity_inputs:
            print("velocity.shape", velocity.shape)
            # out, (hidden_state, cell_state) = self.lstm(velocity, (hidden_state, cell_state))
            hidden_state, cell_state = self.lstm(velocity, (hidden_state, cell_state))

            # print("out.shape", out.shape)
            print("cell_state.shape", cell_state.shape)
            print("hidden_state.shape", hidden_state.shape)
            print("")

        # inputs = torch.cat(velocity_inputs).view(len(velocity_inputs), 10, -1)
        # print("inputs.shape", inputs.shape)
        # out, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))


        # print("LSTM output shape", out.shape)
        print("LSTM hidden shape", hidden_state.shape)
        print("LSTM cell state shape", cell_state.shape)
        # TODO: fix things up so it works in a time distributed fashion properly

        # features = self.linear(out)
        features = self.linear(hidden_state)

        pc_pred = self.pc_output(features)

        hd_pred = self.hd_output(features)

        return pc_pred, hd_pred


model = PathIntegrationModel()

# Binary cross-entropy loss
criterion = nn.BCELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size
)

for epoch in range(n_epochs):

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

        if pc_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        # pc_pred, hd_pred = model(velocity_inputs[i, :, :], pc_inputs[i, :], hd_inputs[i, :])
        pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

        # loss = criterion(pc_pred, pc_outputs[i, :]) + criterion(hd_pred, hd_outputs[i, :])
        loss = criterion(pc_pred, pc_outputs) + criterion(hd_pred, hd_outputs)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data.item()
        n_batches += 1



