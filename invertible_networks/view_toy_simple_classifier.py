import sys
import torch
import torch.nn as nn
from models import FeedForward
from toy_dataset import ToyDataset, plot_data
import matplotlib.pyplot as plt
import numpy as np


fname = sys.argv[1]

n_samples = 10000
hidden_size = 512

model = FeedForward(input_size=2, hidden_size=hidden_size, output_size=4)
model.load_state_dict(torch.load(fname), strict=True)
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

        outputs = model(locations)

        loss = criterion(outputs, labels)

        print(loss.data.item())

fig, ax = plt.subplots()

loc = locations.detach().numpy()
# convert the one-hot output to a single label
lab = np.argmax(outputs.detach().numpy(), axis=1)

plot_data(loc, lab, ax)

plt.show()
