import torch
import torch.nn as nn
from ssp_navigation.utils.models import FeedForward
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import json
from datetime import datetime
import os.path as osp


def add_training_params(parser):

    parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam', 'sgd'])

    parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
    parser.add_argument('--n-samples', type=int, default=20000,
                        help='Number of samples to generate if a dataset is not given')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val-period', type=int, default=5, help='number of epochs before a test/validation set run')

    return parser


def train(args, trainloader, testloader, input_size, output_size=2):

    model = FeedForward(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size)

    # Open a tensorboard writer if a logging directory is given
    if args.logdir != '':
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_dir = osp.join(args.logdir, current_time)
        writer = SummaryWriter(log_dir=save_dir)
        # if args.weight_histogram:
        #     # Log the initial parameters
        #     for name, param in model.named_parameters():
        #         writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

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

        if e % args.val_period == 0:
            with torch.no_grad():

                # Everything is in one batch, so this loop will only happen once
                for i, data in enumerate(testloader):
                    ssp, coord = data

                    outputs = model(ssp)

                    loss = criterion(outputs, coord)

                if args.logdir != '':

                    if output_size == 2:
                        fig_pred, ax_pred = plt.subplots()
                        plot_predictions_v(
                            predictions=outputs, coords=coord,
                            ax=ax_pred,
                            min_val=-args.limit*1.1,
                            max_val=args.limit*1.1,
                            fixed_axes=False,
                        )
                        writer.add_figure('test set predictions', fig_pred, e)
                    writer.add_scalar('test_loss', loss.data.item(), e)

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

        if args.logdir != '':
            if n_batches > 0:
                avg_loss /= n_batches
                writer.add_scalar('avg_loss', avg_loss, e + 1)

            # if args.weight_histogram and (e + 1) % 10 == 0:
            #     for name, param in model.named_parameters():
            #         writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

    print("Testing")
    with torch.no_grad():

        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):

            ssp, coord = data

            outputs = model(ssp)

            loss = criterion(outputs, coord)

            # print(loss.data.item())

        if args.logdir != '':

            if output_size == 2:
                fig_pred, ax_pred = plt.subplots()
                fig_truth, ax_truth = plt.subplots()
                plot_predictions_v(
                    predictions=outputs, coords=coord,
                    ax=ax_pred,
                    min_val=-args.limit*1.1,
                    max_val=args.limit*1.1,
                    fixed_axes=False,
                )
                writer.add_figure('test set predictions', fig_pred, args.epochs)
                plot_predictions_v(
                    predictions=coord, coords=coord,
                    ax=ax_truth,
                    min_val=-args.limit*1.1,
                    max_val=args.limit*1.1,
                    fixed_axes=False,
                )
                writer.add_figure('ground truth', fig_truth)
            # fig_hist = plot_histogram(predictions=outputs, coords=coord)
            # writer.add_figure('test set histogram', fig_hist)
            writer.add_scalar('test_loss', loss.data.item(), args.epochs)

    # Close tensorboard writer
    if args.logdir != '':
        writer.close()

        torch.save(model.state_dict(), osp.join(save_dir, 'model.pt'))

        params = vars(args)
        # # Additionally save the axis vectors used
        # params['x_axis_vec'] = list(x_axis_sp.v)
        # params['y_axis_vec'] = list(y_axis_sp.v)
        with open(osp.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f)
