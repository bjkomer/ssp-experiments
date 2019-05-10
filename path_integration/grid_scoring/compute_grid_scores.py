# Compute grid scores for rate maps using deepmind's code for consistency
import scores
import utils
from run_network import run_and_gather_activations
import numpy as np

import argparse

parser = argparse.ArgumentParser('Compute grid scores for a path integration model')
parser.add_argument('--n-samples', type=int, default=5000)

args = parser.parse_args()

fname_pred = 'sac_{}samples_pred.pdf'.format(args.n_samples)
fname_truth = 'sac_{}samples_truth.pdf'.format(args.n_samples)

ssp_scaling = 5

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
latest_epoch_scorer = scores.GridScorer(
    nbins=20,
    coords_range=((0, 2.2), (0, 2.2)), #data_reader.get_coord_range(),
    mask_parameters=masks_parameters,
)

activations, predictions, coords = run_and_gather_activations(
    n_samples=args.n_samples
)

predictions = predictions / ssp_scaling
coords = coords / ssp_scaling

print(np.max(predictions))
print(np.min(predictions))
# assert False


# grid_scores['btln_60'], grid_scores['btln_90'], \
# grid_scores['btln_60_separation'], grid_scores['btln_90_separation'] = utils.get_scores_and_plot(
grid_scores_60, grid_scores_90, grid_scores_60_separation, grid_scores_90_separation = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=predictions, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_scores', #FLAGS.saver_results_directory,
    filename=fname_pred,
)

grid_scores_60, grid_scores_90, grid_scores_60_separation, grid_scores_90_separation = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=coords, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_scores', #FLAGS.saver_results_directory,
    filename=fname_truth,
)


print(grid_scores_60, grid_scores_90, grid_scores_60_separation, grid_scores_90_separation)