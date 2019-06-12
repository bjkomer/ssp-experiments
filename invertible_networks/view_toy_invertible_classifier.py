import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import InvertibleBlock, InvertibleNetwork
from toy_dataset import ToyDataset, plot_data
import matplotlib.pyplot as plt
import numpy as np
import json
import os

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

hidden_sizes = [hidden_size] * n_hidden_layers
model = InvertibleNetwork(
    input_output_size=max(input_size, output_size),
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

with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        locations, labels = data

        # pad locations with zeros to match label dimensionality
        locations_padded = F.pad(locations, pad=(0, 2), mode='constant', value=0)

        outputs = model(locations_padded)

        loss = criterion(outputs, labels)

        print(loss.data.item())

fig, ax = plt.subplots(3)

loc = locations.detach().numpy()
# convert the one-hot output to a single label
lab = np.argmax(outputs.detach().numpy(), axis=1)

# Forward classification
plot_data(loc, lab, ax[0])

# Backward generation

location_generation = model.backward(outputs)
# remove the padded dimensions
location_trimmed = location_generation.detach().numpy()[:, :2]

plot_data(location_trimmed, lab, ax[1])
# ax[1].set_xlim([-4, 4])
# ax[1].set_ylim([-4, 4])

# Backward generation with harder labels
one_hot_labels = np.zeros((n_samples, 4))
for i in range(n_samples):
    one_hot_labels[i, lab[i]] = 1
one_hot_labels = torch.Tensor(one_hot_labels)
# softmax_outputs = F.softmax(outputs)

# location_generation = model.backward(softmax_outputs)
location_generation = model.backward(one_hot_labels)
# remove the padded dimensions
location_trimmed = location_generation.detach().numpy()[:, :2]

plot_data(location_trimmed, lab, ax[2])

plt.show()
