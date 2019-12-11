import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
# NOTE: this is currently soft-linked to this directory
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from ssp_navigation.utils.encodings import get_encoding_function
from ssp_navigation.utils.models import MLP, LearnedEncoding

from utils import create_train_test_image_dataloaders, PolicyValidationSet
import nengo.spa as spa

parser = argparse.ArgumentParser(
    'Train a network on an image construction task using a specified location encoding'
)

# parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train for')
parser.add_argument('--epoch-offset', type=int, default=0,
                    help='Optional offset to start epochs counting from. To be used when continuing training')
parser.add_argument('--viz-period', type=int, default=50, help='number of epochs before a viz set run')
parser.add_argument('--val-period', type=int, default=25, help='number of epochs before a test/validation set run')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding'
                    ])
                    # choices=['ssp', '2d', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq'])
parser.add_argument('--logdir', type=str, default='output/ssp_image_construction',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--dataset', type=str, default='data/image_dataset.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='mse',
                    choices=['mse', 'cosine', 'combined', 'alternating', 'scaled'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--n-hidden-layers', type=int, default=1)
parser.add_argument('--subsample', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=50000, help='Number of testing samples')

parser.add_argument('--n-images', type=int, default=20, help='Number of different images to use from the dataset')
parser.add_argument('--fixed-goals', action='store_true', help='Goal is always get to (0,0). Used for debugging')

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.01)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)

parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--id-dim', type=int, default=512, help='Dimensionality of the ID vector')
parser.add_argument('--train-split', type=float, default=0.8, help='Training fraction of the train/test split')
parser.add_argument('--allow-cache', action='store_true',
                    help='once the dataset has been generated, it will be saved to a file to be loaded faster')

parser.add_argument('--learning-rate', type=float, default=1e-5, help='Step size multiplier in the RMSProp algorithm')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")


args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

data = np.load(args.dataset)

# images is n_images by size by size by channels

image_size = data['images'].shape[1]

limit_low = 0
limit_high = image_size / 2
res = 128 #256

encoding_func, dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)
# dim = args.encoding_dim

# xs = np.linspace(limit_low, limit_high, res)
# ys = np.linspace(limit_low, limit_high, res)
#
# # FIXME: inefficient but will work for now
# heatmap_vectors = np.zeros((len(xs), len(ys), dim))
#
# print("Generating Heatmap Vectors")
#
# for i, x in enumerate(xs):
#     for j, y in enumerate(ys):
#         heatmap_vectors[i, j, :] = encoding_func(
#             x=x, y=y
#         )
#
#         heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])
#
# print("Heatmap Vector Generation Complete")

# # n_samples = 5000
# n_samples = args.n_samples#1000
# rollout_length = args.trajectory_length#100
# batch_size = args.minibatch_size#10
# n_epochs = args.n_epochs#20
# # n_epochs = 5
#
# if args.n_hd_cells > 0:
#     hd_encoding_func = hd_gauss_encoding_func(dim=args.n_hd_cells, sigma=0.25, use_softmax=False, rng=np.random.RandomState(args.seed))
#     if args.sin_cos_ang:
#         input_size = 3
#     else:
#         input_size = 2
#     model = SSPPathIntegrationModel(
#         input_size=input_size, unroll_length=rollout_length,
#         sp_dim=dim + args.n_hd_cells, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
#     )
# else:
#     hd_encoding_func = None
#     model = SSPPathIntegrationModel(
#         input_size=2,
#         unroll_length=rollout_length,
#         sp_dim=dim, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
#     )
#


if 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=2,#repr_dim,
        encoding_size=args.dim,
        maze_id_size=args.id_dim,
        hidden_size=args.hidden_size,
        output_size=3,  # RGB
        n_layers=args.n_hidden_layers
    )
else:
    model = MLP(
        input_size=args.id_dim + dim * 2,
        hidden_size=args.hidden_size,
        output_size=3,  # RBG
        n_layers=args.n_hidden_layers
    )

model.to(device)


if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)


# trying out cosine similarity here as well, it works better than MSE as a loss for SSP cleanup
cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

# # encoding specific cache string
# encoding_specific = ''
# if args.spatial_encoding == 'ssp':
#     encoding_specific = args.ssp_scaling
# elif args.spatial_encoding == 'frozen-learned':
#     encoding_specific = args.frozen_model
# elif args.spatial_encoding == 'pc-gauss' or args.spatial_encoding == 'pc-gauss-softmax':
#     encoding_specific = args.pc_gauss_sigma
# elif args.spatial_encoding == 'hex-trig':
#     encoding_specific = args.hex_freq_coef
#
# cache_fname = 'dataset_cache/{}_{}_{}_{}_{}_{}.npz'.format(
#     args.spatial_encoding, args.dim, args.seed, args.n_samples, args.n_hd_cells, encoding_specific
# )
#
# # if the file exists, load it from cache
# if os.path.exists(cache_fname):
#     print("Generating Train and Test Loaders from Cache")
#     trainloader, testloader = load_from_cache(cache_fname, batch_size=batch_size, n_samples=n_samples)
# else:
#     print("Generating Train and Test Loaders")
#
#     if args.n_hd_cells > 0:
#         trainloader, testloader = angular_train_test_loaders(
#             data,
#             n_train_samples=n_samples,
#             n_test_samples=n_samples,
#             rollout_length=rollout_length,
#             batch_size=batch_size,
#             encoding=args.spatial_encoding,
#             encoding_func=encoding_func,
#             encoding_dim=args.dim,
#             train_split=args.train_split,
#             hd_dim=args.n_hd_cells,
#             hd_encoding_func=hd_encoding_func,
#             sin_cos_ang=args.sin_cos_ang,
#         )
#     else:
#         trainloader, testloader = train_test_loaders(
#             data,
#             n_train_samples=n_samples,
#             n_test_samples=n_samples,
#             rollout_length=rollout_length,
#             batch_size=batch_size,
#             encoding=args.spatial_encoding,
#             encoding_func=encoding_func,
#             encoding_dim=args.dim,
#             train_split=args.train_split,
#         )
#
#     if args.allow_cache:
#
#         if not os.path.exists('dataset_cache'):
#             os.makedirs('dataset_cache')
#
#         np.savez(
#             cache_fname,
#             train_velocity_inputs=trainloader.dataset.velocity_inputs,
#             train_ssp_inputs=trainloader.dataset.ssp_inputs,
#             train_ssp_outputs=trainloader.dataset.ssp_outputs,
#             test_velocity_inputs=testloader.dataset.velocity_inputs,
#             test_ssp_inputs=testloader.dataset.ssp_inputs,
#             test_ssp_outputs=testloader.dataset.ssp_outputs,
#         )
#
# print("Train and Test Loaders Generation Complete")

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

# Keep track of running average losses, to adaptively scale the weight between them
running_avg_cosine_loss = 1.
running_avg_mse_loss = 1.

id_vecs = np.zeros((args.n_images, args.dim))
# overwrite data
for i in range(args.n_images):
    id_vecs[i, :] = spa.SemanticPointer(args.dim).v


if args.n_images < 4:
    image_indices = list(np.arange(args.n_images))
    n_goals = 4
else:
    image_indices = [0, 1, 2, 3]
    n_goals = 2


# TODO: make a version of this for images
validation_set = PolicyValidationSet(
    data=data, dim=args.dim, id_vecs=id_vecs, image_indices=image_indices, n_goals=n_goals, subsample=args.subsample,
    # spatial_encoding=args.spatial_encoding,
    encoding_func=encoding_func, device=device,
    fixed_goals=args.fixed_goals,
    seed=13
)


trainloader, testloader = create_train_test_image_dataloaders(
    data=data, n_train_samples=args.n_train_samples, n_test_samples=args.n_test_samples,
    id_vecs=id_vecs, args=args, n_images=args.n_images,
    encoding_func=encoding_func, pin_memory=pin_memory,
    fixed_goals=args.fixed_goals
)

validation_set.run_ground_truth(writer=writer)


for e in range(args.epoch_offset, args.epochs + args.epoch_offset):
    print('Epoch: {0}'.format(e + 1))

    if e % args.viz_period == 0:
        print("Running Viz Set")
        # do a validation run and save images
        validation_set.run_validation(model, writer, e)

        if e > 0:
            # Save a copy of the model at this stage
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_epoch_{}.pt'.format(e)))

    # Run the test set for validation
    if e % args.val_period == 0:
        print("Running Val Set")
        avg_test_mse_loss = 0
        avg_test_cosine_loss = 0
        n_test_batches = 0
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(device))

                mse_loss = mse_criterion(outputs, directions.to(device))
                cosine_loss = cosine_criterion(
                    outputs,
                    directions.to(device),
                    torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
                )

                avg_test_mse_loss += mse_loss.data.item()
                avg_test_cosine_loss += cosine_loss.data.item()
                n_test_batches += 1

        if n_test_batches > 0:
            avg_test_mse_loss /= n_test_batches
            avg_test_cosine_loss /= n_test_batches
            print(avg_test_mse_loss, avg_test_cosine_loss)
            writer.add_scalar('test_mse_loss', avg_test_mse_loss, e)
            writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, e)

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if maze_loc_goal_ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(args.batch_size).to(device)
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

    if args.logdir != '':
        if n_batches > 0:
            avg_mse_loss /= n_batches
            avg_cosine_loss /= n_batches
            print(avg_mse_loss, avg_cosine_loss)
            writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)
            writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)


print("Testing")
avg_test_mse_loss = 0
avg_test_cosine_loss = 0
n_test_batches = 0
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
        )

        avg_test_mse_loss += mse_loss.data.item()
        avg_test_cosine_loss += cosine_loss.data.item()
        n_test_batches += 1

if n_test_batches > 0:
    avg_test_mse_loss /= n_test_batches
    avg_test_cosine_loss /= n_test_batches
    print(avg_test_mse_loss, avg_test_cosine_loss)
    writer.add_scalar('test_mse_loss', avg_test_mse_loss, args.epochs + args.epoch_offset)
    writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, args.epochs + args.epoch_offset)


print("Visualization")
validation_set.run_validation(model, writer, args.epochs + args.epoch_offset)


# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)

    params['id_vecs'] = [list(id_vecs[i, :]) for i in range(args.n_images)]

    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)






































# print("Training")
# for epoch in range(args.n_epochs):
#     # print('\x1b[2K\r Epoch {} of {}'.format(epoch + 1, n_epochs), end="\r")
#     print('Epoch {} of {}'.format(epoch + 1, args.n_epochs))
#
#     # TODO: modularize this and clean it up
#     # Every 'eval_period' epochs, create a test loss and image
#     if epoch % args.eval_period == 0:
#         print('Evaluating at Epoch {}'.format(epoch + 1))
#         with torch.no_grad():
#             # Everything is in one batch, so this loop will only happen once
#             for i, data in enumerate(testloader):
#                 velocity_inputs, ssp_inputs, ssp_outputs = data
#
#                 ssp_pred = model(velocity_inputs, ssp_inputs)
#
#                 # NOTE: need to permute axes of the targets here because the output is
#                 #       (sequence length, batch, units) instead of (batch, sequence_length, units)
#                 #       could also permute the outputs instead
#                 cosine_loss = cosine_criterion(
#                     ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#                     ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#                     torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
#                 )
#                 mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))
#
#                 # TODO: handle loss differently for HD version
#
#                 print("test cosine loss", cosine_loss.data.item())
#                 print("test mse loss", mse_loss.data.item())
#                 writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)
#                 writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
#                 writer.add_scalar('test_combined_loss', mse_loss.data.item() + cosine_loss.data.item(), epoch)
#                 c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#                 m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#                 writer.add_scalar(
#                     'test_scaled_loss',
#                     mse_loss.data.item() * m_f + cosine_loss.data.item() * c_f,
#                     epoch
#                 )
#
#             # print("ssp_pred.shape", ssp_pred.shape)
#             # print("ssp_outputs.shape", ssp_outputs.shape)
#
#             # Just use start and end location to save on memory and computation
#             predictions_start = np.zeros((ssp_pred.shape[1], 2))
#             coords_start = np.zeros((ssp_pred.shape[1], 2))
#
#             predictions_end = np.zeros((ssp_pred.shape[1], 2))
#             coords_end = np.zeros((ssp_pred.shape[1], 2))
#
#             if args.spatial_encoding == 'ssp':
#                 print("computing prediction locations")
#                 predictions_start[:, :] = ssp_to_loc_v(
#                     ssp_pred.detach().numpy()[0, :, :args.dim],
#                     heatmap_vectors, xs, ys
#                 )
#                 predictions_end[:, :] = ssp_to_loc_v(
#                     ssp_pred.detach().numpy()[-1, :, :args.dim],
#                     heatmap_vectors, xs, ys
#                 )
#                 print("computing ground truth locations")
#                 coords_start[:, :] = ssp_to_loc_v(
#                     ssp_outputs.detach().numpy()[:, 0, :args.dim],
#                     heatmap_vectors, xs, ys
#                 )
#                 coords_end[:, :] = ssp_to_loc_v(
#                     ssp_outputs.detach().numpy()[:, -1, :args.dim],
#                     heatmap_vectors, xs, ys
#                 )
#             elif args.spatial_encoding == '2d':
#                 print("copying prediction locations")
#                 predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
#                 predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
#                 print("copying ground truth locations")
#                 coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
#                 coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]
#             else:
#                 # normalizing is important here
#                 print("computing prediction locations")
#                 pred_start = ssp_pred.detach().numpy()[0, :, :args.dim]
#                 # pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
#                 predictions_start[:, :] = ssp_to_loc_v(
#                     pred_start,
#                     heatmap_vectors, xs, ys
#                 )
#                 pred_end = ssp_pred.detach().numpy()[-1, :, :args.dim]
#                 # pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
#                 predictions_end[:, :] = ssp_to_loc_v(
#                     pred_end,
#                     heatmap_vectors, xs, ys
#                 )
#                 print("computing ground truth locations")
#                 coord_start = ssp_outputs.detach().numpy()[:, 0, :args.dim]
#                 # coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
#                 coords_start[:, :] = ssp_to_loc_v(
#                     coord_start,
#                     heatmap_vectors, xs, ys
#                 )
#                 coord_end = ssp_outputs.detach().numpy()[:, -1, :args.dim]
#                 # coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
#                 coords_end[:, :] = ssp_to_loc_v(
#                     coord_end,
#                     heatmap_vectors, xs, ys
#                 )
#
#             fig_pred_start, ax_pred_start = plt.subplots()
#             fig_truth_start, ax_truth_start = plt.subplots()
#             fig_pred_end, ax_pred_end = plt.subplots()
#             fig_truth_end, ax_truth_end = plt.subplots()
#
#             print("plotting predicted locations")
#             plot_predictions_v(predictions_start, coords_start, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
#             plot_predictions_v(predictions_end, coords_end, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
#             print("plotting ground truth locations")
#             plot_predictions_v(coords_start, coords_start, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
#             plot_predictions_v(coords_end, coords_end, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)
#
#             writer.add_figure("predictions start", fig_pred_start, epoch)
#             writer.add_figure("ground truth start", fig_truth_start, epoch)
#
#             writer.add_figure("predictions end", fig_pred_end, epoch)
#             writer.add_figure("ground truth end", fig_truth_end, epoch)
#
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(save_dir, '{}_path_integration_model_epoch_{}.pt'.format(args.spatial_encoding, epoch))
#             )
#
#     avg_bce_loss = 0
#     avg_cosine_loss = 0
#     avg_mse_loss = 0
#     avg_combined_loss = 0
#     avg_scaled_loss = 0
#     n_batches = 0
#     for i, data in enumerate(trainloader):
#         velocity_inputs, ssp_inputs, ssp_outputs = data
#
#         if ssp_inputs.size()[0] != args.batch_size:
#             continue  # Drop data, not enough for a batch
#         optimizer.zero_grad()
#         # model.zero_grad()
#
#         ssp_pred = model(velocity_inputs, ssp_inputs)
#
#         # NOTE: need to permute axes of the targets here because the output is
#         #       (sequence length, batch, units) instead of (batch, sequence_length, units)
#         #       could also permute the outputs instead
#         cosine_loss = cosine_criterion(
#             ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#             ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#             torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
#         )
#         mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))
#         loss = cosine_loss + mse_loss
#
#         # adaptive weighted combination of the two loss functions
#         c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#         m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#         scaled_loss = cosine_loss * c_f + mse_loss * m_f
#
#         if args.loss_function == 'cosine':
#             cosine_loss.backward()
#         elif args.loss_function == 'mse':
#             mse_loss.backward()
#         elif args.loss_function == 'combined':
#             loss.backward()
#         elif args.loss_function == 'alternating':
#             if epoch % 2 == 0:
#                 cosine_loss.backward()
#             else:
#                 mse_loss.backward()
#         elif args.loss_function == 'scaled':
#             scaled_loss.backward()
#
#         avg_cosine_loss += cosine_loss.data.item()
#         avg_mse_loss += mse_loss.data.item()
#         avg_combined_loss += (cosine_loss.data.item() + mse_loss.data.item())
#         avg_scaled_loss += scaled_loss.data.item()
#
#         # Gradient Clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)
#
#         optimizer.step()
#
#         # avg_loss += loss.data.item()
#         n_batches += 1
#
#     avg_cosine_loss /= n_batches
#     avg_mse_loss /= n_batches
#     avg_combined_loss /= n_batches
#     avg_scaled_loss /= n_batches
#     print("cosine loss:", avg_cosine_loss)
#     print("mse loss:", avg_mse_loss)
#     print("combined loss:", avg_combined_loss)
#     print("scaled loss:", avg_scaled_loss)
#
#     running_avg_cosine_loss = 0.9 * running_avg_cosine_loss + 0.1 * avg_cosine_loss
#     running_avg_mse_loss = 0.9 * running_avg_mse_loss + 0.1 * avg_mse_loss
#     print("running_avg_cosine_loss", running_avg_cosine_loss)
#     print("running_avg_mse_loss", running_avg_mse_loss)
#
#     writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)
#     writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
#     writer.add_scalar('avg_combined_loss', avg_combined_loss, epoch + 1)
#     writer.add_scalar('avg_scaled_loss', avg_scaled_loss, epoch + 1)
#
#
# print("Testing")
# with torch.no_grad():
#     # Everything is in one batch, so this loop will only happen once
#     for i, data in enumerate(testloader):
#         velocity_inputs, ssp_inputs, ssp_outputs = data
#
#         ssp_pred = model(velocity_inputs, ssp_inputs)
#
#         # NOTE: need to permute axes of the targets here because the output is
#         #       (sequence length, batch, units) instead of (batch, sequence_length, units)
#         #       could also permute the outputs instead
#         cosine_loss = cosine_criterion(
#             ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#             ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
#             torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
#         )
#         mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))
#
#         print("final test cosine loss", cosine_loss.data.item())
#         print("final test mse loss", mse_loss.data.item())
#         writer.add_scalar('final_test_cosine_loss', cosine_loss.data.item(), epoch)
#         writer.add_scalar('final_test_mse_loss', mse_loss.data.item(), epoch)
#         writer.add_scalar('final_test_combined_loss', mse_loss.data.item() + cosine_loss.data.item(), epoch)
#         c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#         m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
#         writer.add_scalar(
#             'final_test_scaled_loss',
#             mse_loss.data.item() * m_f + cosine_loss.data.item() * c_f,
#             epoch
#         )
#
#     # Just use start and end location to save on memory and computation
#     predictions_start = np.zeros((ssp_pred.shape[1], 2))
#     coords_start = np.zeros((ssp_pred.shape[1], 2))
#
#     predictions_end = np.zeros((ssp_pred.shape[1], 2))
#     coords_end = np.zeros((ssp_pred.shape[1], 2))
#
#     if args.spatial_encoding == 'ssp':
#         print("computing prediction locations")
#         predictions_start[:, :] = ssp_to_loc_v(
#             ssp_pred.detach().numpy()[0, :, :args.dim],
#             heatmap_vectors, xs, ys
#         )
#         predictions_end[:, :] = ssp_to_loc_v(
#             ssp_pred.detach().numpy()[-1, :, :args.dim],
#             heatmap_vectors, xs, ys
#         )
#         print("computing ground truth locations")
#         coords_start[:, :] = ssp_to_loc_v(
#             ssp_outputs.detach().numpy()[:, 0, :args.dim],
#             heatmap_vectors, xs, ys
#         )
#         coords_end[:, :] = ssp_to_loc_v(
#             ssp_outputs.detach().numpy()[:, -1, :args.dim],
#             heatmap_vectors, xs, ys
#         )
#     elif args.spatial_encoding == '2d':
#         print("copying prediction locations")
#         predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
#         predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
#         print("copying ground truth locations")
#         coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
#         coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]
#     else:
#         # normalizing is important here
#         print("computing prediction locations")
#         pred_start = ssp_pred.detach().numpy()[0, :, :args.dim]
#         # pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
#         predictions_start[:, :] = ssp_to_loc_v(
#             pred_start,
#             heatmap_vectors, xs, ys
#         )
#         pred_end = ssp_pred.detach().numpy()[-1, :, :args.dim]
#         # pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
#         predictions_end[:, :] = ssp_to_loc_v(
#             pred_end,
#             heatmap_vectors, xs, ys
#         )
#         print("computing ground truth locations")
#         coord_start = ssp_outputs.detach().numpy()[:, 0, :args.dim]
#         # coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
#         coords_start[:, :] = ssp_to_loc_v(
#             coord_start,
#             heatmap_vectors, xs, ys
#         )
#         coord_end = ssp_outputs.detach().numpy()[:, -1, :args.dim]
#         # coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
#         coords_end[:, :] = ssp_to_loc_v(
#             coord_end,
#             heatmap_vectors, xs, ys
#         )
#
#     fig_pred_start, ax_pred_start = plt.subplots()
#     fig_truth_start, ax_truth_start = plt.subplots()
#     fig_pred_end, ax_pred_end = plt.subplots()
#     fig_truth_end, ax_truth_end = plt.subplots()
#
#     print("plotting predicted locations")
#     plot_predictions_v(predictions_start, coords_start, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
#     plot_predictions_v(predictions_end, coords_end, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
#     print("plotting ground truth locations")
#     plot_predictions_v(coords_start, coords_start, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
#     plot_predictions_v(coords_end, coords_end, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)
#
#     writer.add_figure("final predictions start", fig_pred_start)
#     writer.add_figure("final ground truth start", fig_truth_start)
#
#     writer.add_figure("final predictions end", fig_pred_end)
#     writer.add_figure("final ground truth end", fig_truth_end)
#
# torch.save(model.state_dict(), os.path.join(save_dir, '{}_path_integration_model.pt'.format(args.spatial_encoding)))
