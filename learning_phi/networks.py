import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.autograd import Function
# from jax import grad, vmap, jacfwd, jacrev
# from jax.ops import index, index_update
# import jax.numpy as jnp


class SSPTransform(nn.Module):

    def __init__(self, coord_dim, ssp_dim):
        super(SSPTransform, self).__init__()

        # dimensionality of the input coordinates
        self.coord_dim = coord_dim

        # dimensionality of the SSP
        self.ssp_dim = ssp_dim

        # number of phi parameters to learn
        self.n_param = (ssp_dim-1) // 2

        # self.phis = nn.Parameter(torch.ones(self.n_param, dtype=torch.complex64))
        # self.phis = nn.Parameter(torch.Tensor(self.n_param, dtype=torch.complex64))
        self.phis = nn.Parameter(torch.Tensor(self.coord_dim, self.n_param))

        # initialize parameters
        torch.nn.init.uniform_(self.phis, a=-np.pi + 0.001, b=np.pi - 0.001)

        # number of phis, plus constant, plus potential nyquist if even
        self.tau_len = (self.ssp_dim // 2) + 1
        # constants used in the transformation
        # first dimension is batch dimension, set to 1 to be broadcastable
        self.const_phase = torch.zeros(1, self.ssp_dim, self.tau_len)
        for a in range(self.ssp_dim):
            for k in range(self.tau_len):
                self.const_phase[:, a, k] = 2*np.pi*k*a/self.ssp_dim

        # The 2/N or 1/N scaling applied outside of the cos
        # 2/N on all terms with phi, 1/N on constant and nyquist if it exists
        self.const_scaling = torch.ones(1, 1, self.tau_len)*2./self.ssp_dim
        self.const_scaling[:, :, 0] = 1./self.ssp_dim
        if self.ssp_dim % 2 == 0:
            self.const_scaling[:, :, -1] = 1. / self.ssp_dim


        # if self.coord_dim == 1:
        #     self.ssp_func = SSPFunction.apply
        # else:
        #     self.ssp_func = SSPFunction2D.apply

    def forward(self, inputs):

        # return self.ssp_func(inputs, self.phis[0,:])

        batch_size = inputs.shape[0]

        full_phis = torch.zeros(self.coord_dim, self.tau_len, dtype=inputs.dtype)
        full_phis[:, 1:self.n_param+1] = self.phis
        shift = torch.zeros(batch_size, 1, self.tau_len, dtype=inputs.dtype)
        shift[:, 0, :] = torch.mm(inputs, full_phis)

        # shift[:, :, 1:self.n_param+1]
        # shift[:, :, 0] = 0.5
        #
        # # Nyquist term for even dimension
        # if self.ssp_dim % 2 == 0:
        #     shift[:, :, -1] = 0.5

        return (torch.cos(shift + self.const_phase)*self.const_scaling).sum(axis=2)

        # ret = torch.ones(batch_size, self.ssp_dim, dtype=torch.float64) * 1. / self.ssp_dim
        # for ri in range(self.ssp_dim):
        #     for k in range(self.n_param):
        #         if self.coord_dim == 1:
        #             ret[:, ri] += (2./self.ssp_dim) * torch.cos(
        #                 self.phis[0, k]*inputs[:, 0] + 2*np.pi*(k+1)*ri/self.ssp_dim
        #             )
        #         elif self.coord_dim == 2:
        #             ret[:, ri] += (2. / self.ssp_dim) * torch.cos(
        #                 self.phis[0, k] * inputs[:, 0] + self.phis[1, k] * inputs[:, 1] + 2 * np.pi * (k + 1) * ri / self.ssp_dim
        #             )
        #         else:
        #             raise NotImplementedError
        #         # Nyquist term for even dimension
        #         if self.ssp_dim % 2 == 0:
        #             ret[:, ri] += (-1)**(k+1)/self.ssp_dim
        #
        # return ret

        # return self.ssp_func(inputs, self.phis)

        # batch_size = inputs.shape[0]
        #
        # const_term = torch.ones(batch_size, 2)
        # const_term[:, 1] = 0
        #
        # # rot_f = torch.pow(self.phis, inputs)
        # # rot_f = torch.pow(torch.exp(1.j * self.phis), inputs)
        # rot_f = np.exp(1.j * self.phis)** inputs
        #
        # print(rot_f.shape)
        #
        # # prepend the constant term of 1 before doing irfft
        # return torch.irfft(
        #     # torch.cat([torch.ones(batch_size, 1), rot_f]),
        #     torch.cat([const_term, rot_f]),
        #     signal_ndim=1, normalized=True, onesided=True,
        #     signal_sizes=(self.dim,)  # signal sizes is without batch dimension
        # )


class BasicDataset(data.Dataset):

    def __init__(self, inputs, outputs):

        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):

        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return self.inputs.shape[0]


def get_train_test_loaders(encode_func, limit=1, rng=np.random, input_dim=1, output_dim=3, n_train_samples=1000, n_test_samples=1000, batch_size=32):
    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        # inputs = np.zeros((n_samples, input_dim))
        inputs = rng.uniform(-limit, limit, size=(n_samples, input_dim))
        outputs = np.zeros((n_samples, output_dim))

        for n in range(n_samples):

            outputs[n, :] = encode_func(inputs[n, :])

            dataset = BasicDataset(
                inputs=inputs,
                outputs=outputs,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0
            )

    return trainloader, testloader


# def encode_point_even(x, phis):
#     # dim = len(phis)*2 + 2
#     dim = phis.shape[0]*1 + 2
#     uf = jnp.zeros((dim,), dtype='complex64')
#     uf = index_update(uf, index[0], 1)
#     uf = index_update(uf, index[1:-1], jnp.exp(1.j*phis))
#     uf = index_update(uf, index[-1], 1)
#     # jax version of irfft assumes there is a nyquist frequency
#     # they have not implemented it for odd dimensions
#     ret = jnp.fft.irfft(uf**x)
#     return ret
#
#
# def encode_point_odd(x, phis):
#     # dim = len(phis)*2 + 1
#     dim = phis.shape[0]*2 + 1
#     uf = jnp.zeros((dim,), dtype='complex64')
#     uf = index_update(uf, index[0], 1)
#     uf = index_update(uf, index[1:(dim+1)//2], jnp.exp(1.j*phis))
#     uf = index_update(uf, index[-1:dim//2:-1], jnp.conj(jnp.exp(1.j*phis)))
#     # this version uses full ifft with complex to allow odd dim
#     ret = jnp.fft.ifft(uf**x).real
#     return ret
#
#
# # second argument to vmap indicates that the phis do not have a batch
# batch_encode_point_even = vmap(encode_point_even, (0, None))
# grad_encode_point_even_x = vmap(jacrev(encode_point_even, argnums=0), (0, None))
# grad_encode_point_even_phis = vmap(jacrev(encode_point_even, argnums=1), (0, None))
# batch_encode_point_odd = vmap(encode_point_odd, (0, None))
# grad_encode_point_odd_x = vmap(jacrev(encode_point_odd, argnums=0), (0, None))
# grad_encode_point_odd_phis = vmap(jacrev(encode_point_odd, argnums=1), (0, None))
#
#
# # Inherit from Function
# class SSPFunction(Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     def forward(ctx, input, phis):
#         input_np = input.detach().numpy()
#         phis_np = phis.detach().numpy()
#
#         # print(input_np.shape)
#         # print(phis_np.shape)
#
#         ret = np.asarray(batch_encode_point_odd(input_np, phis_np))
#         # ret = np.asarray(batch_encode_point_even(input_np, phis_np))
#
#         # dim = len(phis) + 1
#         # uf = np.zeros((dim, ), dtype='complex64')
#         # uf[0] = 1
#         # uf[1:] = np.exp(1.j*phis_np)
#         # ret = np.fft.irfft(uf ** input_np, n=dim)
#         ctx.save_for_backward(input, phis)
#
#         return torch.DoubleTensor(ret)
#
#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, phis = ctx.saved_tensors
#         grad_input = grad_phis = None
#
#         input_np = input.detach().numpy()
#         phis_np = phis.detach().numpy()
#         ret_x = torch.DoubleTensor(np.asarray(grad_encode_point_odd_x(input_np, phis_np)))
#         ret_phis = torch.DoubleTensor(np.asarray(grad_encode_point_odd_phis(input_np, phis_np)))
#         # ret_x = torch.DoubleTensor(np.asarray(grad_encode_point_even_x(input_np, phis_np)))
#         # ret_phis = torch.DoubleTensor(np.asarray(grad_encode_point_even_phis(input_np, phis_np)))
#
#         # print(grad_output)
#         # print(ret_x)
#         # print(ret_phis)
#         # print("")
#
#         # print("grad_output.shape", grad_output.shape)
#         # print("ret_x.shape", ret_x.shape)
#         # print("ret_phis.shape", ret_phis.shape)
#
#         # step by step gradient computation to make sure it is correct
#         # should be able to vectorize in the future
#
#         # set up output shapes
#         grad_input = torch.zeros(ret_x.shape[0], ret_x.shape[2])
#         grad_phis = torch.zeros(ret_phis.shape[0], ret_phis.shape[2])
#
#         # each batch should be treated independently, so explicity do that
#
#         batch_size = ret_x.shape[0]
#         for bi in range(batch_size):
#             # matrix multiply to remove the output dimension
#             # print(grad_input[bi, :].shape)
#             # print("=")
#             # print(ret_x[bi, :, :].shape)
#             # print("X")
#             # print(grad_output[bi, :].shape)
#             # print("")
#             grad_input[bi, :] = ret_x[bi, :, :].t() @ grad_output[bi, :]
#             grad_phis[bi, :] = ret_phis[bi, :, :].t() @ grad_output[bi, :]
#
#         # sum across the batch dimension to get the update for the phis
#         grad_phis = grad_phis.sum(0)
#
#         # print("grad shapes")
#         # print(grad_input.shape)
#         # print(grad_phis.shape)
#         # print(grad_input)
#         # print(grad_phis)
#
#         return grad_input, grad_phis
#
#     # # This function has only a single output, so it gets only one gradient
#     # @staticmethod
#     # def backward(ctx, grad_output):
#     #     # This is a pattern that is very convenient - at the top of backward
#     #     # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#     #     # None. Thanks to the fact that additional trailing Nones are
#     #     # ignored, the return statement is simple even when the function has
#     #     # optional inputs.
#     #     input, phis = ctx.saved_tensors
#     #     grad_input = grad_phis = None
#     #
#     #     input_np = input.detach().numpy()
#     #     phis_np = phis.detach().numpy()
#     #     ret_x = torch.DoubleTensor(np.asarray(grad_encode_point_odd_x(input_np, phis_np)))
#     #     ret_phis = torch.DoubleTensor(np.asarray(grad_encode_point_odd_phis(input_np, phis_np)))
#     #     # ret_x = torch.DoubleTensor(np.asarray(grad_encode_point_even_x(input_np, phis_np)))
#     #     # ret_phis = torch.DoubleTensor(np.asarray(grad_encode_point_even_phis(input_np, phis_np)))
#     #
#     #     # print(grad_output)
#     #     # print(ret_x)
#     #     # print(ret_phis)
#     #     # print("")
#     #
#     #     # print("grad_output.shape", grad_output.shape)
#     #     # print("ret_x.shape", ret_x.shape)
#     #     # print("ret_phis.shape", ret_phis.shape)
#     #
#     #     # step by step gradient computation to make sure it is correct
#     #     # should be able to vectorize in the future
#     #
#     #     # set up output shapes
#     #     grad_input = torch.zeros(ret_x.shape[0], ret_x.shape[2])
#     #     grad_phis = torch.zeros(ret_phis.shape[0], ret_phis.shape[2])
#     #
#     #     # each batch should be treated independently, so explicity do that
#     #
#     #     batch_size = ret_x.shape[0]
#     #     for bi in range(batch_size):
#     #         # matrix multiply to remove the output dimension
#     #         # print(grad_input[bi, :].shape)
#     #         # print("=")
#     #         # print(ret_x[bi, :, :].shape)
#     #         # print("X")
#     #         # print(grad_output[bi, :].shape)
#     #         # print("")
#     #         grad_input[bi, :] = ret_x[bi, :, :].t() @ grad_output[bi, :]
#     #         grad_phis[bi, :] = ret_phis[bi, :, :].t() @ grad_output[bi, :]
#     #
#     #     # sum across the batch dimension to get the update for the phis
#     #     grad_phis = grad_phis.sum(0)
#     #
#     #     # print("grad shapes")
#     #     # print(grad_input.shape)
#     #     # print(grad_phis.shape)
#     #     # print(grad_input)
#     #     # print(grad_phis)
#     #
#     #     return grad_input, grad_phis
#
#
#
# def encode_2d_point_even(pos, phis):
#     dim = phis.shape[1] + 2
#     xf = jnp.zeros((dim,), dtype='complex64')
#     xf = index_update(xf, index[0], 1)
#     xf = index_update(xf, index[1:-1], jnp.exp(1.j*phis[0, :]))
#     xf = index_update(xf, index[-1], 1)
#
#     yf = jnp.zeros((dim,), dtype='complex64')
#     yf = index_update(yf, index[0], 1)
#     yf = index_update(yf, index[1:-1], jnp.exp(1.j*phis[1, :]))
#     yf = index_update(yf, index[-1], 1)
#     # jax version of irfft assumes there is a nyquist frequency
#     # they have not implemented it for odd dimensions
#     ret = jnp.fft.irfft(xf**pos[0]*yf**pos[1])
#     return ret
#
#
# # second argument to vmap indicates that the phis do not have a batch
# batch_encode_2d_point_even = vmap(encode_2d_point_even, (0, None))
# grad_encode_2d_point_even_x = vmap(jacrev(encode_2d_point_even, argnums=0), (0, None))
# grad_encode_2d_point_even_phis = vmap(jacrev(encode_2d_point_even, argnums=1), (0, None))
# # batch_encode_2d_point_odd = vmap(encode_2d_point_odd, (0, None))
# # grad_encode_2d_point_odd_x = vmap(jacrev(encode_2d_point_odd, argnums=0), (0, None))
# # grad_encode_2d_point_odd_phis = vmap(jacrev(encode_2d_point_odd, argnums=1), (0, None))
#
#
# # Inherit from Function
# class SSPFunction2D(Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     def forward(ctx, input, phis):
#         input_np = input.detach().numpy()
#         phis_np = phis.detach().numpy()
#
#         # print(input_np.shape)
#         # print(phis_np.shape)
#
#         ret = np.asarray(batch_encode_2d_point_even(input_np, phis_np))
#
#         ctx.save_for_backward(input, phis)
#
#         return torch.DoubleTensor(ret)
#
#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, phis = ctx.saved_tensors
#         grad_input = grad_phis = None
#
#         input_np = input.detach().numpy()
#         phis_np = phis.detach().numpy()
#         ret_x = torch.DoubleTensor(np.asarray(grad_encode_2d_point_even_x(input_np, phis_np)))
#         ret_phis = torch.DoubleTensor(np.asarray(grad_encode_2d_point_even_phis(input_np, phis_np)))
#
#         # print(grad_output)
#         # print(ret_x)
#         # print(ret_phis)
#         # print("")
#
#         # print("grad_output.shape", grad_output.shape)
#         # print("ret_x.shape", ret_x.shape)
#         # print("ret_phis.shape", ret_phis.shape)
#
#         # step by step gradient computation to make sure it is correct
#         # should be able to vectorize in the future
#
#         # set up output shapes
#         grad_input = torch.zeros(ret_x.shape[0], ret_x.shape[2])
#         grad_phis = torch.zeros(ret_phis.shape[0], ret_phis.shape[2], ret_phis.shape[3])
#
#         # each batch should be treated independently, so explicity do that
#
#         batch_size = ret_x.shape[0]
#         for bi in range(batch_size):
#             # matrix multiply to remove the output dimension
#             # print(grad_input[bi, :].shape)
#             # print("=")
#             # print(ret_x[bi, :, :].shape)
#             # print("X")
#             # print(grad_output[bi, :].shape)
#             # print("")
#             # print(grad_phis[bi, :].shape)
#             # print("=")
#             # print(ret_phis[bi, :, :].shape)
#             # print("X")
#             # print(grad_output[bi, :].shape)
#             # print("")
#             grad_input[bi, :] = ret_x[bi, :, :].t() @ grad_output[bi, :]
#             # loop through x and y phis
#             for pi in range(2):
#                 grad_phis[bi, pi, :] = ret_phis[bi, :, pi, :].t() @ grad_output[bi, :]
#
#         # sum across the batch dimension to get the update for the phis
#         grad_phis = grad_phis.sum(0)
#
#         # print("grad shapes")
#         # print(grad_input.shape)
#         # print(grad_phis.shape)
#         # print(grad_input)
#         # print(grad_phis)
#
#         return grad_input, grad_phis


def test_ssp_op():
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    # input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),
    #          torch.randn(30, 20, dtype=torch.double, requires_grad=True))
    # test = gradcheck(linear, input, eps=1e-6, atol=1e-4)

    ssp_func = SSPFunction.apply

    batch_size = 1 # 20

    input = (torch.randn(batch_size, 1, dtype=torch.double, requires_grad=True),
             torch.randn(1, dtype=torch.double, requires_grad=True))
    input = (torch.ones(batch_size, 1, dtype=torch.double, requires_grad=True),
             torch.ones(1, dtype=torch.double, requires_grad=True))
    print(input)
    test = gradcheck(ssp_func, input, eps=1e-6, atol=1e-4)

    print(test)


if __name__ == '__main__':
    test_ssp_op()
