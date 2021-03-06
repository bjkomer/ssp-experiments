import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from functools import partial

from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay


def cconv_f(a, bf):
    af = np.fft.fft(a)
    return np.fft.ifft(af*bf).real


class CircConvPathIntegrationModel(object):

    def __init__(self, x_axis_vec, y_axis_vec, unroll_length=100, dt=0.02, scaling=0.5):

        self.x_axis_vec = x_axis_vec
        self.y_axis_vec = y_axis_vec
        self.xf = np.fft.fft(self.x_axis_vec)
        self.yf = np.fft.fft(self.y_axis_vec)
        self.unroll_length = unroll_length
        self.dim = len(self.x_axis_vec)
        self.dt = dt
        # self.dt = dt/scaling
        # self.dt = dt*scaling

    def forward(self, velocity_inputs, initial_ssp):

        batch_size = velocity_inputs[0].shape[0]

        output = np.zeros((self.unroll_length+1, batch_size, self.dim))

        # vel = velocity_inputs.detach().numpy()
        init_ssp = initial_ssp.detach().numpy()

        # TODO: improve efficiency by vectorizing
        for bi in range(batch_size):
            # output[0, bi, :] = initial_ssp[bi, :]
            output[0, bi, :] = init_ssp[bi, :]
            for ri in range(self.unroll_length):
                # vel_ssp = encode_point(velocity_inputs[ri][bi, 0], velocity_inputs[ri][bi, 1])
                # vel_f = (self.xf**velocity_inputs[ri][bi, 0])*(self.yf**velocity_inputs[ri][bi, 1])
                # vel_f = (self.xf ** vel[ri][bi, 0]) * (self.yf ** vel[ri][bi, 1])
                vel = velocity_inputs[ri].detach().numpy()*self.dt
                vel_f = (self.xf ** vel[bi, 0]) * (self.yf ** vel[bi, 1])
                output[ri + 1, bi, :] = cconv_f(output[ri, bi, :], vel_f)

        return torch.Tensor(output[1:, :, :])


class Simple2DPathIntegrationModel(object):

    def __init__(self, unroll_length=100, dt=0.02):
        self.unroll_length = unroll_length
        self.dt = dt

    def forward(self, velocity_inputs, initial_ssp):

        batch_size = velocity_inputs[0].shape[0]

        output = np.zeros((self.unroll_length + 1, batch_size, 2))

        # vel = velocity_inputs.detach().numpy()
        init_ssp = initial_ssp.detach().numpy()

        # TODO: improve efficiency by vectorizing
        for bi in range(batch_size):
            # output[0, bi, :] = initial_ssp[bi, :]
            output[0, bi, :] = init_ssp[bi, :]
            for ri in range(self.unroll_length):
                vel = velocity_inputs[ri].detach().numpy()*self.dt
                output[ri + 1, bi, :] = output[ri, bi, :] + vel[bi, :]

        return torch.Tensor(output[1:, :, :])


class SSPPathIntegrationModel(nn.Module):

    def __init__(self, input_size=2, lstm_hidden_size=128, linear_hidden_size=512,
                 unroll_length=100, sp_dim=512, dropout_p=.5, use_lmu=False, order=6):

        super(SSPPathIntegrationModel, self).__init__()

        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.unroll_length = unroll_length
        self.use_lmu = use_lmu
        self.order = order

        self.sp_dim = sp_dim

        if use_lmu:
            self.lmu_cell = LMUCell(
                input_size=self.input_size,
                hidden_size=self.lstm_hidden_size,
                order=order,
            )
            # memory size is the same as the order
            n_cell = order
        else:
            # Full LSTM that can be given the full sequence and produce the full output in one step
            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=1
            )
            # cell state size set to hidden state size to be consistent
            n_cell = self.lstm_hidden_size

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
            out_features=n_cell,
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

        if self.use_lmu:
            # h = hidden_state.view(1, batch_size, self.lstm_hidden_size)
            # c = cell_state.view(1, batch_size, self.lstm_hidden_size)
            h = hidden_state.view(batch_size, self.lstm_hidden_size)
            c = cell_state.view(batch_size, self.order)
            outputs = []
            for velocity in velocity_inputs:
                h, c = self.lmu_cell(velocity, (h, c))
                outputs.append(h)
            output = torch.cat(outputs).view(len(outputs), batch_size, -1)
        else:
            velocities = torch.cat(velocity_inputs).view(len(velocity_inputs), batch_size, -1)
            # Note that output is a sequence, it contains a vector for every timestep
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

        if self.use_lmu:
            # h = hidden_state.view(1, batch_size, self.lstm_hidden_size)
            # c = cell_state.view(1, batch_size, self.lstm_hidden_size)
            h = hidden_state.view(batch_size, self.lstm_hidden_size)
            c = cell_state.view(batch_size, self.order)
            outputs = []
            for velocity in velocity_inputs:
                h, c = self.lmu_cell(velocity, (h, c))
                outputs.append(h)
            output = torch.cat(outputs).view(len(outputs), batch_size, -1)
        else:
            velocities = torch.cat(velocity_inputs).view(len(velocity_inputs), batch_size, -1)
            # Note that output is a sequence, it contains a vector for every timestep
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

        return ssp_pred, output, features


# from: https://github.com/deepsound-project/samplernn-pytorch/blob/master/nn.py#L46
def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))


# based on: https://github.com/abr/neurips2019/blob/master/lmu/lmu.py
# class LMUCell(nn.modules.rnn.RNNCellBase):
class LMUCell(nn.Module):

    def __init__(self, input_size, hidden_size, #bias=True,
                 order,
                 theta=100,  # relative to dt=1
                 method='zoh',
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer=lecun_uniform,
                 hidden_encoders_initializer=lecun_uniform,
                 memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
                 input_kernel_initializer=torch.nn.init.xavier_normal_,
                 hidden_kernel_initializer=torch.nn.init.xavier_normal_,
                 memory_kernel_initializer=torch.nn.init.xavier_normal_,
                 hidden_activation='tanh',
                 ):
        super(LMUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.order = order
        # self.bias = bias

        if hidden_activation == 'tanh':
            self.hidden_activation = torch.tanh
        elif hidden_activation == 'relu':
            self.hidden_activation = torch.relu

        realizer = Identity()
        self._realizer_result = realizer(
            LegendreDelay(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1., method=method)
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        # TODO: might need some transposing of sizes
        self.input_encoders = nn.Parameter(torch.Tensor(1, input_size), requires_grad=trainable_input_encoders)
        self.hidden_encoders = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=trainable_hidden_encoders)
        self.memory_encoders = nn.Parameter(torch.Tensor(1, order), requires_grad=trainable_memory_encoders)
        self.input_kernel = nn.Parameter(torch.Tensor(hidden_size, input_size), requires_grad=trainable_input_kernel)
        self.hidden_kernel = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=trainable_hidden_kernel)
        self.memory_kernel = nn.Parameter(torch.Tensor(hidden_size, order), requires_grad=trainable_memory_kernel)
        self.AT = nn.Parameter(torch.Tensor(self._A), requires_grad=trainable_A)
        self.BT = nn.Parameter(torch.Tensor(self._B), requires_grad=trainable_B)

        # Initialize parameters
        input_encoders_initializer(self.input_encoders)
        hidden_encoders_initializer(self.hidden_encoders)
        memory_encoders_initializer(self.memory_encoders)
        input_kernel_initializer(self.input_kernel)
        hidden_kernel_initializer(self.hidden_kernel)
        memory_kernel_initializer(self.memory_kernel)

    def forward(self, input, hx):

        h, m = hx

        u = (F.linear(input, self.input_encoders) +
             F.linear(h, self.hidden_encoders) +
             F.linear(m, self.memory_encoders))

        m = m + F.linear(m, self.AT) + F.linear(u, self.BT)

        h = self.hidden_activation(
            F.linear(input, self.input_kernel) +
            F.linear(h, self.hidden_kernel) +
            F.linear(m, self.memory_kernel))

        return h, m
