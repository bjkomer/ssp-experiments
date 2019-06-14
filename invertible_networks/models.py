import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np


class FeedForward(nn.Module):
    """
    Single hidden layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

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


# modified from: https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n].contiguous()
    x2 = x[:, n:].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class InvertibleBlock(nn.Module):

    def __init__(self, input_output_size, hidden_size=512):
        super(InvertibleBlock, self).__init__()
        assert input_output_size % 2 == 0
        dim = input_output_size // 2
        self.s1 = FeedForward(input_size=dim, hidden_size=hidden_size, output_size=dim)
        self.s2 = FeedForward(input_size=dim, hidden_size=hidden_size, output_size=dim)
        self.t1 = FeedForward(input_size=dim, hidden_size=hidden_size, output_size=dim)
        self.t2 = FeedForward(input_size=dim, hidden_size=hidden_size, output_size=dim)

        self.z_dist = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def forward(self, input):
        u1, u2 = split(input)

        # TEMP FIXME
        # log_det_J = input.new_zeros(input.shape[0])

        v1 = u1 * torch.exp(self.s2(u2)) + self.t2(u2)
        v2 = u2 * torch.exp(self.s1(v1)) + self.t1(v1)
        # s1 = self.s1(v1)
        # v2 = u2 * torch.exp(s1) + self.t1(v1)

        # log_det_J -= (v1.sum(dim=1) + v2.sum(dim=1))
        # log_det_J -= v1.sum(dim=1)
        # log_det_J -= s1.sum(dim=1)

        output = merge(v1, v2)

        return output
        # return output, log_det_J

    def backward(self, output):
        v1, v2 = split(output)

        u2 = (v2 - self.t1(v1)) * torch.exp(-self.s1(v1))
        u1 = (v1 - self.t2(u2)) * torch.exp(-self.s2(u2))

        input = merge(u1, u2)

        return input


class InvertibleNetwork(nn.Module):

    def __init__(self, input_output_size, hidden_sizes=(512, 512), z_dim=2):
        super(InvertibleNetwork, self).__init__()

        # self.blocks = []
        #
        # for hidden_size in hidden_sizes:
        #     self.blocks.append(
        #         InvertibleBlock(input_output_size=input_output_size, hidden_size=hidden_size)
        #     )
        # self.modules = nn.ModuleList(self.blocks)

        self.blocks = nn.ModuleList(
            InvertibleBlock(
                input_output_size=input_output_size, hidden_size=hidden_size
            ) for hidden_size in hidden_sizes
        )

        self.z_dim = z_dim
        self.z_dist = distributions.MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))

    def forward(self, input):

        # TEMP FIXME
        # log_det_J = input.new_zeros(input.shape[0])

        output = input
        for block in self.blocks:
            output = block.forward(output)
            # output, logp = block.forward(output)
            # log_det_J += logp
        return output
        # return output, log_det_J

    def backward(self, output):

        input = output
        for block in self.blocks[::-1]:
            input = block.backward(input)
        return input


# from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):

    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


def inverse_multiquadratic(x1, x2, h):
    # TODO: what is h?
    return 1 / (1 + np.linalg.norm((x1 - x2) / h)**2)


def inverse_multiquadratic_v2(x1, x2, C=1):
    # C is: 2 * d_z * sigma**2,
    # which is the expected squared distance between two multivariate Gaussian vectors drawn from P_Z
    # return C / (C + np.linalg.norm((x1 - x2))**2)
    return C / (C + torch.norm((x1 - x2))**2)
