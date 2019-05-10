import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
# NOTE: this is currently soft-linked to this directory
from arguments import add_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
from localization_training_utils import TrajectoryValidationSet, localization_train_test_loaders, LocalizationModel, pc_to_loc_v


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with trajectories using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d', 'pc'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--logdir', type=str, default='output/ssp_trajectory_localization',
                    help='Directory for saved model and tensorboard log')
# TODO: update default to use dataset with distance sensor measurements (or boundary cell activations)
parser.add_argument('--dataset', type=str, default='../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='cosine', choices=['cosine', 'mse'])

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cosine_loss = False
if args.loss_function == 'cosine':
    use_cosine_loss = True

if use_cosine_loss:
    print("Using Cosine Loss")
else:
    print("Using MSE Loss")

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']

pc_centers = data['pc_centers']
pc_activations = data['pc_activations']

n_sensors = data['n_sensors']

if args.encoding == 'ssp':
    dim = 512
elif args.encoding == '2d':
    dim = 2
    ssp_scaling = 1  # no scaling used for 2D coordinates directly
elif args.encoding == 'pc':
    dim = args.n_place_cells
    ssp_scaling = 1
else:
    raise NotImplementedError

limit_low = 0 * ssp_scaling
limit_high = 2.2 * ssp_scaling
res = 128 #256

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

# Input is x and y velocity plus the distance sensor measurements
model = LocalizationModel(
    input_size=2 + n_sensors,
    unroll_length=rollout_length,
    sp_dim=dim
)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

if args.encoding == 'pc':
    # Binary cross-entropy loss
    # criterion = nn.BCELoss()
    # more numerically stable. Do not use softmax beforehand
    criterion = nn.BCEWithLogitsLoss()
else:
    cosine_criterion = nn.CosineEmbeddingLoss()
    mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = localization_train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size,
    encoding=args.encoding,
)

validation_set = TrajectoryValidationSet(
    dataloader=testloader,
    heatmap_vectors=heatmap_vectors,
    xs=xs,
    ys=ys,
    ssp_scaling=ssp_scaling,
    spatial_encoding=args.encoding,
)

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:

        print("Evaluating")
        validation_set.run_eval(
            model=model,
            writer=writer,
            epoch=epoch,
        )

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, sensor_inputs, ssp_inputs, ssp_outputs = data

        if ssp_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        ssp_pred = model(velocity_inputs, sensor_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        if args.encoding == 'pc':
            # place cell version needs to explicitly do the softmax here
            loss = criterion(ssp_pred, F.softmax(ssp_outputs.permute(1, 0, 2), dim=2))
        else:
            cosine_loss = cosine_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2),
                             torch.ones(ssp_pred.shape[0], ssp_pred.shape[1]))
            mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

        if use_cosine_loss:
            cosine_loss.backward()
        else:
            mse_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

    avg_mse_loss /= n_batches
    avg_cosine_loss /= n_batches
    print("mse loss:", avg_mse_loss)
    print("cosine loss:", avg_cosine_loss)
    writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
    writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)


print("Testing")
validation_set.run_eval(
    model=model,
    writer=writer,
    epoch=n_epochs,
)

if args.encoding == 'ssp':
    torch.save(model.state_dict(), os.path.join(save_dir, 'ssp_trajectory_localization_model.pt'))
elif args.encoding == '2d':
    torch.save(model.state_dict(), os.path.join(save_dir, '2d_trajectory_localization_model.pt'))
