import torch.nn as nn
import torch.nn.functional as F
import torch


class SSPPathIntegrationModel(nn.Module):

    def __init__(self, input_size=2, lstm_hidden_size=128, linear_hidden_size=512,
                 unroll_length=100, sp_dim=512, dropout_p=.5):

        super(SSPPathIntegrationModel, self).__init__()

        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.unroll_length = unroll_length

        self.sp_dim = sp_dim

        # Full LSTM that can be given the full sequence and produce the full output in one step
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1
        )

        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=self.linear_hidden_size,
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.ssp_output = nn.Linear(
            in_features=self.linear_hidden_size,
            out_features=self.sp_dim
        )

        # Linear transforms for ground truth ssp into initial hidden and cell state of lstm
        self.w_c = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_h = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

    def forward(self, velocity_inputs, initial_ssp):

        batch_size = velocity_inputs[0].shape[0]

        # Compute initial hidden state
        cell_state = self.w_c(initial_ssp)
        hidden_state = self.w_h(initial_ssp)

        velocities = torch.cat(velocity_inputs).view(len(velocity_inputs), batch_size, -1)

        output, (_, _) = self.lstm(
            velocities,
            (
                hidden_state.view(1, batch_size, self.lstm_hidden_size),
                cell_state.view(1, batch_size, self.lstm_hidden_size)
            )
        )

        features = self.dropout(self.linear(output))

        # TODO: should normalization be used here?
        ssp_pred = self.ssp_output(features)

        return ssp_pred

    def forward_activations(self, velocity_inputs, initial_ssp):
        """Returns the hidden layer activations as well as the prediction"""

        batch_size = velocity_inputs[0].shape[0]

        # Compute initial hidden state
        cell_state = self.w_c(initial_ssp)
        hidden_state = self.w_h(initial_ssp)

        velocities = torch.cat(velocity_inputs).view(len(velocity_inputs), batch_size, -1)

        output, (_, _) = self.lstm(
            velocities,
            (
                hidden_state.view(1, batch_size, self.lstm_hidden_size),
                cell_state.view(1, batch_size, self.lstm_hidden_size)
            )
        )

        features = self.dropout(self.linear(output))

        # TODO: should normalization be used here?
        ssp_pred = self.ssp_output(features)

        return ssp_pred, output
