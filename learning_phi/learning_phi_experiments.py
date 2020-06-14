import numpy as np
import torch.nn as nn
import torch
from networks import SSPTransform, get_train_test_loaders
import os
import argparse


parser = argparse.ArgumentParser('Experiment with learning phis with pytorch')
parser.add_argument('--n-epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine', 'combined'])
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--n-train-samples', type=int, default=10000)
parser.add_argument('--n-test-samples', type=int, default=10000)
parser.add_argument('--ssp-dim', type=int, default=5)
parser.add_argument('--coord-dim', type=int, default=2)
parser.add_argument('--limit', type=float, default=1.0)
parser.add_argument('--n-seeds', type=int, default=10)

args = parser.parse_args()


def main(args):
    torch.set_default_dtype(torch.float64)

    for limit in [0.5, 1.0, 2.0, 4.0]:
        for ssp_dim in [5, 10, 15, 20, 100]:
            for coord_dim in [1, 2]:
                args.limit = limit
                args.ssp_dim = ssp_dim
                args.coord_dim = coord_dim

                n_phis = (args.ssp_dim-1)//2

                true_phis = np.zeros((args.n_seeds, n_phis, args.coord_dim))
                learned_phis = np.zeros((args.n_seeds, n_phis, args.coord_dim))
                losses = np.zeros((args.n_seeds, args.n_epochs))
                val_losses = np.zeros((args.n_seeds, args.n_epochs))

                for seed in range(args.n_seeds):
                    true_phi, learned_phi, loss, val_loss = experiment(
                        seed=seed,
                        limit=args.limit,
                        ssp_dim=args.ssp_dim,
                        coord_dim=args.coord_dim,
                        batch_size=args.batch_size,
                        n_train_samples=args.n_train_samples,
                        n_test_samples=args.n_test_samples,
                        args=args,
                    )

                    true_phis[seed, :, :] = true_phi
                    learned_phis[seed, :, :] = learned_phi
                    losses[seed, :] = loss
                    val_losses[seed, :] = val_loss

                if not os.path.exists('output'):
                    os.makedirs('output')

                fname = 'output/learned_phi_results_{}D_{}dim_{}limit.npz'.format(args.coord_dim, args.ssp_dim, args.limit)

                np.savez(
                    fname,
                    true_phis=true_phis,
                    learned_phis=learned_phis,
                    losses=losses,
                    val_losses=val_losses,
                )


def experiment(
        args,
        seed, ssp_dim, limit, coord_dim=2,
        batch_size=32, n_train_samples=10000, n_test_samples=10000,

):
    """
    :param seed: seed for generating ssp
    :param ssp_dim: dimensionality of ssp
    :param space_dim: dimensionality of the coord space
    :return:
        true_phi
        learned_phi
        loss_over_time
        val_loss_over_time
    """

    losses = np.zeros((args.n_epochs,))
    val_losses = np.zeros((args.n_epochs,))

    rng = np.random.RandomState(seed=seed)
    n_phis = (ssp_dim-1)//2
    phis = rng.uniform(-np.pi+0.001, np.pi-0.001, size=(coord_dim, n_phis))
    xf = np.ones((n_phis + 2,), dtype='complex64')
    xf[1:-1] = np.exp(1.j * phis[0, :])
    if coord_dim == 2:
        yf = np.ones((n_phis + 2,), dtype='complex64')
        yf[1:-1] = np.exp(1.j * phis[1, :])

        def encode_func(pos):
            return np.fft.irfft((xf ** pos[0]) * (yf ** pos[1]), n=ssp_dim)
    else:
        def encode_func(pos):
            return np.fft.irfft((xf ** pos), n=ssp_dim)

    model = SSPTransform(coord_dim=coord_dim, ssp_dim=ssp_dim)

    trainloader, testloader = get_train_test_loaders(
        encode_func, rng=rng, batch_size=batch_size,
        input_dim=coord_dim, output_dim=ssp_dim,
        n_train_samples=n_train_samples, n_test_samples=n_test_samples,
        limit=limit
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

    print("Training")
    for epoch in range(args.n_epochs):
        avg_mse_loss = 0
        avg_cosine_loss = 0
        n_batches = 0
        for i, data in enumerate(trainloader):
            inputs, outputs = data

            if inputs.size()[0] != batch_size:
                continue  # Drop data, not enough for a batch
            optimizer.zero_grad()

            predictions = model(inputs)

            mse_loss = mse_criterion(predictions, outputs)
            cosine_loss = cosine_criterion(
                predictions,
                outputs,
                torch.ones(batch_size)
            )
            # print(loss.data.item())
            avg_mse_loss += mse_loss.data.item()
            avg_cosine_loss += cosine_loss.data.item()
            n_batches += 1

            if args.loss_function == 'mse':
                mse_loss.backward()
            elif args.loss_function == 'cosine':
                cosine_loss.backward()
            elif args.loss_function == 'combined':
                (mse_loss + cosine_loss).backward()

            optimizer.step()

        losses[epoch] = avg_mse_loss / n_batches
        print(avg_mse_loss / n_batches, avg_cosine_loss / n_batches)

        # Evaluating
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
            model.train()

        avg_test_mse_loss /= n_test_batches
        avg_test_cosine_loss /= n_test_batches
        val_losses[epoch] = avg_test_mse_loss

    return phis, model.phis.detach().numpy(), losses, val_losses


if __name__ == '__main__':
    main(args)
