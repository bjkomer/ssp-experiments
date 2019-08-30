# This version explicitly learns a latent space distribution
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from models import InvertibleBlock, InvertibleNetwork, inverse_multiquadratic_v2
from toy_dataset import ToyDataset, plot_data
import matplotlib.pyplot as plt
import numpy as np
import json
import os

np.random.seed(13)
torch.manual_seed(13)

# should point to the model folder, so both model and params can be read
fname = sys.argv[1]
model_fname = os.path.join(fname, 'model.pt')
params_fname = os.path.join(fname, 'params.json')

with open(params_fname, "r") as f:
    params = json.load(f)

n_samples = 10000
hidden_size = params['hidden_size']
n_hidden_layers = params['n_hidden_layers']

input_size = 2
output_size = 4
latent_size = 2

input_padding = output_size + latent_size - input_size

hidden_sizes = [hidden_size] * n_hidden_layers
model = InvertibleNetwork(
    input_output_size=max(input_size, output_size + latent_size),
    hidden_sizes=hidden_sizes,
)
# model = InvertibleBlock(
#     input_output_size=max(input_size, output_size),
#     hidden_size=hidden_size
# )
model.load_state_dict(torch.load(model_fname), strict=True)
model.eval()

dataset_test = ToyDataset(n_samples)

# For testing just do everything in one giant batch
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
)

criterion = nn.CrossEntropyLoss()

z_dist = distributions.MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))

with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        locations, labels = data

        # pad locations with zeros to match label dimensionality
        locations_padded = F.pad(locations, pad=(0, input_padding), mode='constant', value=0)

        outputs = model(locations_padded)

        output_class = outputs[:, :output_size]
        output_latent = outputs[:, output_size:]

        loss_class = criterion(output_class, labels)

        # TODO: make this correct
        loss_latent = inverse_multiquadratic_v2(
            z_dist.sample((output_latent.shape[0],)),
            output_latent
        )

        loss = loss_class + loss_latent

        print(loss.data.item())

fig, ax = plt.subplots(3)

loc = locations.detach().numpy()
# convert the one-hot output to a single label
# lab = np.argmax(outputs.detach().numpy(), axis=1)
lab = np.argmax(output_class.detach().numpy(), axis=1)

# Forward classification
plot_data(loc, lab, ax[0])

# Backward generation

location_generation = model.backward(outputs)
# remove the padded dimensions
location_trimmed = location_generation.detach().numpy()[:, :2]

plot_data(location_trimmed, lab, ax[1])

# Backward generation with harder labels
one_hot_labels = np.zeros((n_samples, 4))
latent_sample = z_dist.sample((n_samples,)) #np.zeros((n_samples, latent_size))
print(latent_sample.shape)
for i in range(n_samples):
    one_hot_labels[i, lab[i]] = 1

combined = np.hstack([one_hot_labels, latent_sample])
# combined = np.hstack([output_class.detach().numpy(), latent_sample])
# combined = np.hstack([output_class.detach().numpy(), output_latent.detach().numpy()])

assert combined.shape[0] == n_samples
assert combined.shape[1] == output_size + latent_size
one_hot_labels = torch.Tensor(combined)
# one_hot_labels = torch.Tensor(one_hot_labels)
# softmax_outputs = F.softmax(outputs)

# location_generation = model.backward(softmax_outputs)
location_generation = model.backward(one_hot_labels)
# remove the padded dimensions
location_trimmed = location_generation.detach().numpy()[:, :2]

plot_data(location_trimmed, lab, ax[2])

for i in range(3):
    ax[i].set_xlim([-4, 4])
    ax[i].set_ylim([-4, 4])

plt.show()
