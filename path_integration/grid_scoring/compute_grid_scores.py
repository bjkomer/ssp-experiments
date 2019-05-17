# Compute grid scores for rate maps using deepmind's code for consistency
import scores
import utils
from run_network import run_and_gather_activations, run_and_gather_localization_activations
import numpy as np

import argparse

parser = argparse.ArgumentParser('Compute grid scores for a path integration model')
parser.add_argument('--n-samples', type=int, default=5000)
parser.add_argument('--use-localization', action='store_true')
# TODO: use these parameters
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--model', type=str, default='')

args = parser.parse_args()



ssp_scaling = 5

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
latest_epoch_scorer = scores.GridScorer(
    nbins=20,
    coords_range=((0, 2.2), (0, 2.2)), #data_reader.get_coord_range(),
    mask_parameters=masks_parameters,
)

if args.use_localization:
    fname_pred = 'loc_sac_{}samples_pred.pdf'.format(args.n_samples)
    fname_truth = 'loc_sac_{}samples_truth.pdf'.format(args.n_samples)
    # This version has distance sensor measurements as well
    activations, predictions, coords = run_and_gather_localization_activations(
        n_samples=args.n_samples
    )
else:
    # "../../lab/reproducing/data/path_integration_trajectories_logits_1000t_15s_seed13.npz"
    # ../output/ssp_path_integration/ssp_encoding_scaled_loss/gpu3runs/May14_14-31-33/ssp_path_integration_model.pt
    # fname_pred = 'sac_{}samples_pred.pdf'.format(args.n_samples)
    # fname_truth = 'sac_{}samples_truth.pdf'.format(args.n_samples)
    fname_pred = 'scaled_hybrid_sac_{}samples_pred.pdf'.format(args.n_samples)
    fname_truth = 'scaled_hybrid_sac_{}samples_truth.pdf'.format(args.n_samples)
    activations, predictions, coords = run_and_gather_activations(
        n_samples=args.n_samples,
        dataset="../../lab/reproducing/data/path_integration_trajectories_logits_1000t_15s_seed13.npz",
        model_path="../output/ssp_path_integration/ssp_encoding_scaled_loss/gpu3runs/May14_14-31-33/ssp_path_integration_model.pt",
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