import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import argparse
import json
from datetime import datetime
import os.path as osp
from models import InvertibleBlock, InvertibleNetwork
from toy_dataset import ToyDataset


def main():
    parser = argparse.ArgumentParser('Train a simple invertible classifier on a toy dataset')

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the cleanup network')
    parser.add_argument('--n-hidden-layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--logdir', type=str, default='trained_models/invertible_classifier',
                        help='Directory for saved model and tensorboard log')
    parser.add_argument('--load-model', type=str, default='', help='Optional model to continue training from')
    parser.add_argument('--name', type=str, default='',
                        help='Name of output folder within logdir. Will use current date and time if blank')
    parser.add_argument('--weight-histogram', action='store_true', help='Save histograms of the weights if set')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.RandomState(seed=args.seed)

    dataset_train = ToyDataset(args.n_samples)
    dataset_test = ToyDataset(args.n_samples)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # For testing just do everything in one giant batch
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
    )

    input_size = 2
    output_size = 4
    hidden_sizes = [args.hidden_size] * args.n_hidden_layers
    model = InvertibleNetwork(
        input_output_size=max(input_size, output_size),
        hidden_sizes=hidden_sizes,
    )
    # model = InvertibleBlock(
    #     input_output_size=max(input_size, output_size),
    #     hidden_size=args.hidden_size
    # )

    # Open a tensorboard writer if a logging directory is given
    if args.logdir != '':
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_dir = osp.join(args.logdir, current_time)
        writer = SummaryWriter(log_dir=save_dir)
        if args.weight_histogram:
            # Log the initial parameters
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for e in range(args.epochs):
        print('Epoch: {0}'.format(e + 1))

        avg_loss = 0
        n_batches = 0
        for i, data in enumerate(trainloader):

            locations, labels = data

            if locations.size()[0] != args.batch_size:
                continue  # Drop data, not enough for a batch
            optimizer.zero_grad()

            # outputs = torch.max(model(locations), 1)[1].unsqueeze(1)

            # pad locations with zeros to match label dimensionality
            locations = F.pad(locations, pad=(0, 2), mode='constant', value=0)

            outputs = model(locations)

            loss = criterion(outputs, labels)

            avg_loss += loss.data.item()
            n_batches += 1

            loss.backward()

            # print(loss.data.item())

            optimizer.step()

        print(avg_loss / n_batches)

        if args.logdir != '':
            if n_batches > 0:
                avg_loss /= n_batches
                writer.add_scalar('avg_cosine_loss', avg_loss, e + 1)

            if args.weight_histogram and (e + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

    print("Testing")
    with torch.no_grad():

        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):

            locations, labels = data

            # pad locations with zeros to match label dimensionality
            locations = F.pad(locations, pad=(0, 2), mode='constant', value=0)

            outputs = model(locations)

            loss = criterion(outputs, labels)

            print(loss.data.item())

        if args.logdir != '':
            # TODO: get a visualization of the performance
            writer.add_scalar('test_cosine_loss', loss.data.item())

    # Close tensorboard writer
    if args.logdir != '':
        writer.close()

        torch.save(model.state_dict(), osp.join(save_dir, 'model.pt'))

        params = vars(args)
        with open(osp.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f)


if __name__ == '__main__':
    main()
