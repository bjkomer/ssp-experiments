import numpy as np
import torch
from ssp_navigation.utils.models import FeedForward
from ssp_navigation.utils.datasets import GenericDataset
import argparse


parser = argparse.ArgumentParser(
    'runs a pytorch cleanup on the same data that the spiking network sees. Computes output and saved to a file'
)

# these are just for loading data
parser.add_argument('--new-dataset', action='store_true', help='generate a new random dataset to evaluate on')
parser.add_argument('--encoder-type', type=str, default='mixed', choices=['mixed', 'grid', 'band', 'place', 'random'])

parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--n-items', type=int, default=7)

args = parser.parse_args()

# if false, use the test set, wording is weird, need to fix later
args.new_dataset = False

fname = 'pytorch_cleanup_results.npz'

model = FeedForward(input_size=args.dim, hidden_size=args.hidden_size, output_size=args.dim)


model.load_state_dict(torch.load(args.model), strict=False)

# this was tested on the other seed... will need to fix
# cache_fname = 'neural_cleanup_train_dataset_{}.npz'.format(args.dim)
cache_fname = 'neural_cleanup_dataset_{}.npz'.format(args.dim)

cache_data = np.load(cache_fname)
coords = cache_data['coords']
noisy = cache_data['noisy_vectors']
clean_true = cache_data['clean_vectors']

dataset = GenericDataset(inputs=noisy, outputs=clean_true)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=noisy.shape[0], shuffle=False, num_workers=0,
)

for i, data in enumerate(dataloader):

    noisy_input, clean_output = data

    predictions = model(noisy_input)

    pred_vec = predictions.detach().numpy()

np.savez(
    fname,
    pred_vec=pred_vec,
)
