# simple example learning the phi values of an SSP through backpropagation
# most basic example is mapping a line to a circle with a particular scale
# only one phi parameter involved and there is an exact correct setting
import numpy as np
import torch.nn as nn
import torch
from networks import SSPTransform, get_train_test_loaders
import argparse

parser = argparse.ArgumentParser('Simple example learning phis with pytorch')
parser.add_argument('--n-epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine'])
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--n-train-samples', type=int, default=1000)
parser.add_argument('--n-test-samples', type=int, default=1000)
parser.add_argument('--ssp-dim', type=int, default=5)
parser.add_argument('--limit', type=float, default=1.0)
args = parser.parse_args()

# ground truth transform to learn
phi_gt = np.pi/2.
dim = args.ssp_dim
if dim == 4:
    xf = np.ones((dim//2+2, ), dtype='complex64')
    xf[1] = np.exp(1.j * phi_gt)
    yf = np.ones((dim // 2 + 1,), dtype='complex64')
    yf[1] = np.exp(1.j * phi_gt)
    phis = (phi_gt,)
elif dim == 6:
    xf = np.ones((4,), dtype='complex64')
    xf[1] = np.exp(1.j * phi_gt)
    xf[2] = np.exp(1.j * 0)
    yf = np.ones((4,), dtype='complex64')
    yf[1] = np.exp(1.j * 0)
    yf[2] = np.exp(1.j * phi_gt)
    phis = ((phi_gt, 0), (0, phi_gt))
else:
    n_phis = (dim-1)//2
    phis = np.random.uniform(-np.pi+0.001, np.pi-0.001, size=(2, n_phis))
    xf = np.ones((n_phis + 2,), dtype='complex64')
    xf[1:-1] = np.exp(1.j * phis[0, :])
    yf = np.ones((n_phis + 2,), dtype='complex64')
    yf[1:-1] = np.exp(1.j * phis[1, :])
# u_gt_f[2] = np.conj(u_gt_f[1])

# u_gt = np.fft.ifft(u_gt_f).real
# u_gt = np.fft.irfft(u_gt_f[:(dim//2+1)])


def encode_func(pos):
    return np.fft.irfft((xf**pos[0])*(yf**pos[1]), n=dim)


model = SSPTransform(coord_dim=2, ssp_dim=dim)

rng = np.random.RandomState(seed=13)
trainloader, testloader = get_train_test_loaders(
    encode_func, rng=rng, batch_size=args.batch_size,
    input_dim=2, output_dim=dim,
    n_train_samples=args.n_train_samples, n_test_samples=args.n_test_samples,
    limit=args.limit
)


cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError

print("")
print("")
print("correct parameters")
print(phis)
print("")
print("")

print("")
print("Initial parameters:")
print(model.phis)
print("")

print("Training")
for epoch in range(args.n_epochs):
    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        inputs, outputs = data

        if inputs.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        predictions = model(inputs)

        mse_loss = mse_criterion(predictions, outputs)
        cosine_loss = cosine_criterion(
            predictions,
            outputs,
            torch.ones(args.batch_size)
        )
        # print(loss.data.item())
        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

        if args.loss_function == 'mse':
            mse_loss.backward()
        elif args.loss_function == 'cosine':
            cosine_loss.backward()

        optimizer.step()

    print(avg_mse_loss / n_batches, avg_cosine_loss / n_batches)
    print("Current parameters:")
    print(model.phis)
    # if args.logdir != '':
    #     if n_batches > 0:
    #         avg_mse_loss /= n_batches
    #         avg_cosine_loss /= n_batches
    #         print(avg_mse_loss, avg_cosine_loss)
    #         writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)
    #         writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)
    #
    #     if args.weight_histogram and (e + 1) % 10 == 0:
    #         for name, param in model.named_parameters():
    #             writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)


print("Testing")
avg_test_mse_loss = 0
avg_test_cosine_loss = 0
n_test_batches = 0
with torch.no_grad():
    model.eval()
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        inputs, outputs = data

        predictions = model(inputs)

        mse_loss = mse_criterion(predictions, outputs)
        cosine_loss = cosine_criterion(
            predictions,
            outputs,
            torch.ones(inputs.size()[0])
        )

        avg_test_mse_loss += mse_loss.data.item()
        avg_test_cosine_loss += cosine_loss.data.item()
        n_test_batches += 1

if n_test_batches > 0:
    avg_test_mse_loss /= n_test_batches
    avg_test_cosine_loss /= n_test_batches
    print(avg_test_mse_loss, avg_test_cosine_loss)
    # writer.add_scalar('test_mse_loss', avg_test_mse_loss, args.epochs + args.epoch_offset)
    # writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, args.epochs + args.epoch_offset)

print("Learned parameters:")
print(model.phis)
print("correct parameters")
print(phis)
