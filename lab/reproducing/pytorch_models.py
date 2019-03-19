import torch.nn as nn
import torch.nn.functional as F


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

        # print("velocity_inputs[0].shape", velocity_inputs[0].shape)
        # print("initial_pc_activation.shape", initial_pc_activation.shape)
        # print("initial_hd_activation.shape", initial_hd_activation.shape)
        # print("")

        # Compute initial hidden state
        cell_state = self.w_cp(initial_pc_activation) + self.w_cd(initial_hd_activation)
        hidden_state = self.w_hp(initial_pc_activation) + self.w_hd(initial_hd_activation)

        # print("cell_state.shape", cell_state.shape)
        # print("hidden_state.shape", hidden_state.shape)
        # print("")

        # for i in range(self.unroll_length):
        # NOTE: there is a way to do this all at once without a loop:
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # for i in range(self.unroll_length):
        #     print("velocity_inputs[:, i, :].shape", velocity_inputs[:, i, :].shape)
        #     out, (hidden_state, cell_state) = self.lstm(velocity_inputs[:, i, :].view(1, 1, -1), (hidden_state, cell_state))

        for velocity in velocity_inputs:
            # print("velocity.shape", velocity.shape)
            # out, (hidden_state, cell_state) = self.lstm(velocity, (hidden_state, cell_state))
            hidden_state, cell_state = self.lstm(velocity, (hidden_state, cell_state))

            # print("out.shape", out.shape)
            # print("cell_state.shape", cell_state.shape)
            # print("hidden_state.shape", hidden_state.shape)
            # print("")

        # inputs = torch.cat(velocity_inputs).view(len(velocity_inputs), 10, -1)
        # print("inputs.shape", inputs.shape)
        # out, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))


        # print("LSTM output shape", out.shape)
        # print("LSTM hidden shape", hidden_state.shape)
        # print("LSTM cell state shape", cell_state.shape)
        # TODO: fix things up so it works in a time distributed fashion properly

        # features = self.linear(out)
        features = self.linear(hidden_state)

        pc_pred = F.softmax(self.pc_output(features))

        hd_pred = F.softmax(self.hd_output(features))

        return pc_pred, hd_pred

