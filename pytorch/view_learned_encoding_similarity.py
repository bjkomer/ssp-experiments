# view a heatmap of a learned mapping from 2D to 512D for the maze solving task

import numpy as np
import argparse
import torch
from models import LearnedEncoding
import torch.nn.functional as F
import matplotlib.pyplot as plt

res = 128
limit = 5
repr_dim = 2
id_size = 50
model_path = 'multi_maze_solve_function/encodings/50mazes/learned_loc_oh_id/Apr10_16-51-25/model.pt'

model = LearnedEncoding(input_size=repr_dim, maze_id_size=id_size, hidden_size=512, output_size=2)

model.load_state_dict(torch.load(model_path), strict=False)

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

coords = np.zeros((res*res, 2))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        coords[i*len(ys) + j, :] = np.array([xs[i], ys[j]])

#TODO: get the data into a form to easily run through just the first layer of the network

with torch.no_grad():
    # activations = F.relu(model.encoding_layer(coords))
    activations = F.relu(model.encoding_layer(torch.Tensor(coords)))

    print(activations.shape)

    activations_reshaped = activations.detach().numpy().reshape((res, res, 512))

center = activations_reshaped[res // 2, res //2, :]
# center = activations_reshaped[-1, -1, :]
# center = activations_reshaped[32, 64, :]

print(activations_reshaped.shape)
print(center.shape)

heatmap = np.tensordot(activations_reshaped, center, axes=[2, 0])

plt.imshow(heatmap)
# plt.imshow(heatmap, vmin=-1, vmax=1)
plt.show()
