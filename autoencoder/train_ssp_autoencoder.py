import numpy as np
import argparse
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
import os
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from autoencoder_utils import SSPDataset, AutoEncoder


parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--limit', type=int, default=5)
parser.add_argument('--logdir', type=str, default='output')
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--n-train-samples', type=int, default=1000)
parser.add_argument('--n-test-samples', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout fraction')

args = parser.parse_args()

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

trainset = SSPDataset(
    n_samples=args.n_train_samples, limit=args.limit,
    x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
    rng=rng,
)
testset = SSPDataset(
    n_samples=args.n_test_samples, limit=args.limit,
    x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
    rng=rng,
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.n_test_samples, shuffle=False, num_workers=0,
)

model = AutoEncoder(input_dim=args.dim, hidden_dim=args.hidden_size, dropout=args.dropout)

criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

print("Training")
for epoch in range(args.n_epochs):
    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        ssps = data

        if ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        ssp_pred = model(ssps)

        loss = criterion(ssp_pred, ssps)

        loss.backward()

        # Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_loss += loss.data.item()
        n_batches += 1

    avg_loss /= n_batches
    print("loss:", avg_loss)
    writer.add_scalar('avg_loss', avg_loss, epoch + 1)

print("Saving Model")
torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

print("Saving Arguments")
params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

print("Testing")
model.eval()
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        ssps = data

        ssp_pred = model(ssps)

        loss = criterion(ssp_pred, ssps)

        print("test loss", loss.data.item())

    writer.add_scalar('final_test_loss', loss.data.item())
