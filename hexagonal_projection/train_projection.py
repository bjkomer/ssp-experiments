# Very similar setup as training a cleanup memory
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import json
from datetime import datetime
import os.path as osp
import os
# import nengo
import nengo_spa as spa
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions_v
import matplotlib.pyplot as plt
from spatial_semantic_pointers.networks.ssp_cleanup import CoordDecodeDataset, FeedForward
from utils import encode_point_3d, xyz_to_xy_v, xy_to_xyz_v, get_projected_heatmap_vectors

parser = argparse.ArgumentParser('Train a network to project from any 3D SSP to a specific 2D plane')

parser.add_argument('--loss', type=str, default='cosine', choices=['cosine', 'mse'])
# parser.add_argument('--noise-type', type=str, default='memory', choices=['memory', 'gaussian', 'both'])
# parser.add_argument('--sigma', type=float, default=1.0, help='sigma on the gaussian noise if noise-type==gaussian')
parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
parser.add_argument('--n-samples', type=int, default=10000,
                    help='Number of memories to generate. Total samples will be n-samples * n-items')
# parser.add_argument('--n-items', type=int, default=12, help='number of items in memory. Proxy for noisiness')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the cleanup network')
# parser.add_argument('--limits', type=str, default="-5,5,-5,5", help='The limits of the space')
parser.add_argument('--limit', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--logdir', type=str, default='ssp_projection',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--load-model', type=str, default='', help='Optional model to continue training from')
parser.add_argument('--name', type=str, default='',
                    help='Name of output folder within logdir. Will use current date and time if blank')
parser.add_argument('--weight-histogram', action='store_true', help='Save histograms of the weights if set')

args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)


rng = np.random.RandomState(seed=args.seed)
x_axis_sp = make_good_unitary(args.dim, rng=rng)
y_axis_sp = make_good_unitary(args.dim, rng=rng)
z_axis_sp = make_good_unitary(args.dim, rng=rng)


# on the plane
clean_ssps = np.zeros((args.n_samples, args.dim))
# anywhere in 3D space
noisy_ssps = np.zeros((args.n_samples, args.dim))
xy_coords = np.zeros((args.n_samples, 2))
xyz_coords = np.random.uniform(low=-args.limit, high=args.limit, size=(args.n_samples, 3))
xy_coords = xyz_to_xy_v(xyz_coords)
# Represent as 3D coordinates once more, but projected onto the plane
xyz_coords_plane = xy_to_xyz_v(xy_coords)
for i in range(args.n_samples):

    noisy_ssps[i, :] = encode_point_3d(
        x=xyz_coords[i, 0],
        y=xyz_coords[i, 1],
        z=xyz_coords[i, 2],
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp,
        z_axis_sp=z_axis_sp
    ).v

    clean_ssps[i, :] = encode_point_3d(
        x=xyz_coords_plane[i, 0],
        y=xyz_coords_plane[i, 1],
        z=xyz_coords_plane[i, 2],
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp,
        z_axis_sp=z_axis_sp
    ).v

# # Add gaussian noise if required
# if args.noise_type == 'gaussian' or args.noise_type == 'both':
#     noisy_ssps += np.random.normal(loc=0, scale=args.sigma, size=noisy_ssps.shape)

n_samples = clean_ssps.shape[0]
n_train = int(args.train_fraction * n_samples)
n_test = n_samples - n_train
assert(n_train > 0 and n_test > 0)
train_clean = clean_ssps[:n_train, :]
train_noisy = noisy_ssps[:n_train, :]
test_clean = clean_ssps[n_train:, :]
test_noisy = noisy_ssps[n_train:, :]

# NOTE: this dataset is actually generic and can take any input/output mapping
dataset_train = CoordDecodeDataset(vectors=train_noisy, coords=train_clean)
dataset_test = CoordDecodeDataset(vectors=test_noisy, coords=test_clean)

trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
)

# For testing just do everything in one giant batch
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
)

model = FeedForward(dim=dataset_train.dim, hidden_size=args.hidden_size, output_size=dataset_train.dim)

# Open a tensorboard writer if a logging directory is given
if args.logdir != '':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = osp.join(args.logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)
    if args.weight_histogram:
        # Log the initial parameters
        for name, param in model.named_parameters():
            writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

mse_criterion = nn.MSELoss()
cosine_criterion = nn.CosineEmbeddingLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for e in range(args.epochs):
    print('Epoch: {0}'.format(e + 1))

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):

        noisy, clean = data

        if noisy.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(noisy)

        mse_loss = mse_criterion(outputs, clean)
        # Modified to use CosineEmbeddingLoss
        cosine_loss = cosine_criterion(outputs, clean, torch.ones(args.batch_size))

        avg_cosine_loss += cosine_loss.data.item()
        avg_mse_loss += mse_loss.data.item()
        n_batches += 1

        if args.loss == 'cosine':
            cosine_loss.backward()
        else:
            mse_loss.backward()

        # print(loss.data.item())

        optimizer.step()

    print(avg_cosine_loss / n_batches)

    if args.logdir != '':
        if n_batches > 0:
            avg_cosine_loss /= n_batches
            writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)
            writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

print("Testing")
with torch.no_grad():

    for label, loader in zip(['test'], [testloader]):

        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(loader):

            noisy, clean = data

            outputs = model(noisy)

            mse_loss = mse_criterion(outputs, clean)
            # Modified to use CosineEmbeddingLoss
            cosine_loss = cosine_criterion(outputs, clean, torch.ones(len(loader)))

            print(cosine_loss.data.item())

        if args.logdir != '':
            # TODO: get a visualization of the performance

            # show plots of the noisy, clean, and cleaned up with the network
            # note that the plotting mechanism itself uses nearest neighbors, so has a form of cleanup built in

            xs = np.linspace(-args.limit*2, args.limit*2, 256)
            ys = np.linspace(-args.limit*2, args.limit*2, 256)

            heatmap_vectors = get_projected_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp, z_axis_sp)

            noisy_coord = ssp_to_loc_v(
                noisy,
                heatmap_vectors, xs, ys
            )

            pred_coord = ssp_to_loc_v(
                outputs,
                heatmap_vectors, xs, ys
            )

            clean_coord = ssp_to_loc_v(
                clean,
                heatmap_vectors, xs, ys
            )

            fig_noisy_coord, ax_noisy_coord = plt.subplots()
            fig_pred_coord, ax_pred_coord = plt.subplots()
            fig_clean_coord, ax_clean_coord = plt.subplots()

            plot_predictions_v(
                noisy_coord,
                clean_coord,
                ax_noisy_coord, min_val=-args.limit*2, max_val=args.limit*2, fixed_axes=True
            )

            plot_predictions_v(
                pred_coord,
                clean_coord,
                ax_pred_coord, min_val=-args.limit*2, max_val=args.limit*2, fixed_axes=True
            )

            plot_predictions_v(
                clean_coord,
                clean_coord,
                ax_clean_coord, min_val=-args.limit*2, max_val=args.limit*2, fixed_axes=True
            )

            writer.add_figure('{}/original_noise'.format(label), fig_noisy_coord)
            writer.add_figure('{}/test_set_cleanup'.format(label), fig_pred_coord)
            writer.add_figure('{}/ground_truth'.format(label), fig_clean_coord)
            # fig_hist = plot_histogram(predictions=outputs, coords=coord)
            # writer.add_figure('test set histogram', fig_hist)
            writer.add_scalar('{}/test_cosine_loss'.format(label), cosine_loss.data.item())
            writer.add_scalar('{}/test_mse_loss'.format(label), mse_loss.data.item())

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
