import argparse
import numpy as np
from arguments import add_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from trajectory_dataset import train_test_loaders
from pytorch_models import PathIntegrationModel
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import json


parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--logdir', type=str, default='output/pytorch_path_integration',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--dataset', type=str, default='data/path_integration_trajectories_logits_200t_15s.npz')

args = parser.parse_args()

torch.manual_seed(args.seed)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

model = PathIntegrationModel(unroll_length=rollout_length)

# Binary cross-entropy loss
# criterion = nn.BCELoss()
# more numerically stable. Do not use softmax beforehand
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size
)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

        if pc_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        # pc_pred, hd_pred = model(velocity_inputs[i, :, :], pc_inputs[i, :], hd_inputs[i, :])
        pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

        # loss = criterion(pc_pred, pc_outputs[i, :]) + criterion(hd_pred, hd_outputs[i, :])
        loss = criterion(pc_pred, pc_outputs) + criterion(hd_pred, hd_outputs)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data.item()
        n_batches += 1

    avg_loss /= n_batches
    print("loss:", avg_loss)
    writer.add_scalar('avg_loss', avg_loss, epoch + 1)


print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

        pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

        loss = criterion(pc_pred, pc_outputs) + criterion(hd_pred, hd_outputs)

        print("test loss", loss.data.item())

    writer.add_scalar('final_test_loss', loss.data.item())


torch.save(model.state_dict(), os.path.join(save_dir, 'path_integration_model.pt'))

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)
