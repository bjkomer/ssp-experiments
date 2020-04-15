import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
import json
from datetime import datetime
import os.path as osp
from ssp_navigation.utils.encodings import get_encoding_function
from ssp_navigation.utils.datasets import GenericDataset
from ssp_navigation.utils.models import FeedForward
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt


def generate_coord_dataset(encoding_func, n_samples, dim, limit, seed):

    rng = np.random.RandomState(seed=seed)
    coords = rng.uniform(-limit, limit, size=(n_samples, 2))
    vectors = np.zeros((n_samples, dim))

    for n in range(n_samples):
        vectors[n, :] = encoding_func(coords[n, 0], coords[n, 1])

    return vectors, coords


def main():
    parser = argparse.ArgumentParser('Train a network to learn a mapping from an encoded value to 2D coordinate. View output over time.')

    # parser.add_argument('--viz-period', type=int, default=10, help='number of epochs before a viz set run')
    parser.add_argument('--val-period', type=int, default=5, help='number of epochs before a test/validation set run')
    parser.add_argument('--spatial-encoding', type=str, default='hex-ssp',
                        choices=[
                            'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp',
                            'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                            'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                            'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized',
                            'pc-gauss', 'pc-dog', 'tile-coding'
                        ],
                        help='coordinate encoding for agent location and goal')
    parser.add_argument('--freq-limit', type=float, default=10,
                        help='highest frequency of sine wave for random-trig encodings')
    parser.add_argument('--hex-freq-coef', type=float, default=2.5,
                        help='constant to scale frequencies by for hex-trig')
    parser.add_argument('--pc-gauss-sigma', type=float, default=0.75, help='sigma for the gaussians')
    parser.add_argument('--pc-diff-sigma', type=float, default=1.5, help='sigma for subtracted gaussian in DoG')
    parser.add_argument('--hilbert-points', type=int, default=1, choices=[0, 1, 2, 3],
                        help='pc centers. 0: random uniform. 1: hilbert curve. 2: evenly spaced grid. 3: hex grid')
    parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
    parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
    parser.add_argument('--ssp-scaling', type=float, default=1.0)
    parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
    parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
    parser.add_argument('--hidden-size', type=int, default=512)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam', 'sgd'])

    parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
    # NOTE: this is changed to be smaller to see the effects of each epoch more
    parser.add_argument('--n-samples', type=int, default=2500,
                        help='Number of samples to generate if a dataset is not given')
    parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
    parser.add_argument('--limit', type=float, default=1, help='The limits of the space')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--res', type=int, default=64)

    parser.add_argument('--load-model', type=str, default='', help='Optional model to continue training from')

    args = parser.parse_args()

    fname = 'training_over_time_data.npz'

    if osp.exists(fname):
        data = np.load(fname)
        data_outputs = data['data_outputs']
        data_loss = data['data_loss']
        test_coords = data['test_coords']
    else:

        data_outputs = np.zeros((args.epochs, args.res*args.res, 2))
        data_loss = np.zeros((args.epochs,))

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        encoding_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)

        vectors, coords = generate_coord_dataset(
            encoding_func=encoding_func,
            n_samples=args.n_samples,
            dim=repr_dim,
            limit=args.limit,
            seed=args.seed,
        )

        n_samples = vectors.shape[0]
        n_train = int(args.train_fraction * n_samples)
        n_test = n_samples - n_train
        assert(n_train > 0 and n_test > 0)
        train_vectors = vectors[:n_train]
        train_coords = coords[:n_train]
        # test_vectors = vectors[n_train:]
        # test_coords = coords[n_train:]

        test_vectors = np.zeros((args.res*args.res, args.dim))
        test_coords = np.zeros((args.res*args.res, 2))

        # linspace for test, for easy visualization
        xs = np.linspace(-args.limit, args.limit, args.res)
        for i, x in enumerate(xs):
            for j, y in enumerate(xs):
                test_coords[i * args.res + j, 0] = x
                test_coords[i * args.res + j, 1] = y
                test_vectors[i * args.res + j, :] = encoding_func(x, y)

        dataset_train = GenericDataset(inputs=train_vectors, outputs=train_coords)
        dataset_test = GenericDataset(inputs=test_vectors, outputs=test_coords)

        trainloader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )

        # For testing just do everything in one giant batch
        testloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
        )

        model = FeedForward(input_size=repr_dim, hidden_size=args.hidden_size, output_size=2)

        criterion = nn.MSELoss()

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError

        for e in range(args.epochs):
            print('Epoch: {0}'.format(e + 1))


            with torch.no_grad():

                # Everything is in one batch, so this loop will only happen once
                for i, data in enumerate(testloader):
                    ssp, coord = data

                    outputs = model(ssp)

                    loss = criterion(outputs, coord)

                data_loss[e] = loss.data.item()
                data_outputs[e, :, :] = outputs.detach().numpy()


            avg_loss = 0
            n_batches = 0
            for i, data in enumerate(trainloader):

                ssp, coord = data

                if ssp.size()[0] != args.batch_size:
                    continue  # Drop data, not enough for a batch
                optimizer.zero_grad()

                outputs = model(ssp)

                loss = criterion(outputs, coord)
                # print(loss.data.item())
                avg_loss += loss.data.item()
                n_batches += 1

                loss.backward()

                optimizer.step()


        np.savez(
            fname,
            data_outputs=data_outputs,
            data_loss=data_loss,
            test_coords=test_coords,
        )

    # epoch_list = [0, 1, 2, 3, 4, 5]
    # epoch_list = [0, 1, 5, 10, 24]
    epoch_list = [0, 1, 10]
    fix, ax = plt.subplots(1, len(epoch_list), tight_layout=True, figsize=(9, 3))
    # plot data
    for ei, epoch in enumerate(epoch_list):

        plot_predictions_v(
            predictions=data_outputs[ei, :, :], coords=test_coords,
            ax=ax[ei],
            min_val=-args.limit,
            max_val=args.limit,
            fixed_axes=ei != 0,
        )
        ax[ei].set_title("Epoch {} - Loss {:.5f}".format(epoch, data_loss[epoch]))

    plt.show()


if __name__ == '__main__':
    main()
