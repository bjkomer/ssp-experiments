# Compute grid scores for rate maps using deepmind's code for consistency
import scores
import utils
from run_network import run_and_gather_activations, run_and_gather_localization_activations
import numpy as np
# symlinked
from path_integration_utils import encoding_func_from_model

import argparse

parser = argparse.ArgumentParser('Compute grid scores for a path integration model')
parser.add_argument('--n-samples', type=int, default=5000)
parser.add_argument('--use-localization', action='store_true')
# TODO: use these parameters
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--fname-prefix', type=str, default='sac')
parser.add_argument('--ssp-scaling', type=float, default=5.0)
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d', 'pc', 'frozen-learned'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')

args = parser.parse_args()

ssp_scaling = args.ssp_scaling

if args.encoding == 'frozen-learned':
    encoding_func = encoding_func_from_model(args.frozen_model)
else:
    encoding_func = None


if args.use_localization:

    starts = [0.2] * 10
    ends = np.linspace(0.4, 5.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(
        nbins=20,
        coords_range=((0, 10.), (0, 10.)), #data_reader.get_coord_range(),
        mask_parameters=masks_parameters,
    )

else:

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(
        nbins=20,
        coords_range=((0, 2.2), (0, 2.2)), #data_reader.get_coord_range(),
        mask_parameters=masks_parameters,
    )

fname_pred = '{}_{}samples_pred.pdf'.format(args.fname_prefix, args.n_samples)
fname_truth = '{}_{}samples_truth.pdf'.format(args.fname_prefix, args.n_samples)

if args.dataset == '':
    if args.use_localization:
        dataset = '../../localization/data/localization_trajectories_5m_200t_250s_seed13.npz'
    else:
        dataset = '../../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz'
        dataset = "../../lab/reproducing/data/path_integration_trajectories_logits_1000t_15s_seed13.npz"
else:
    dataset = args.dataset

if args.model == '':
    if args.use_localization:
        model = '../../localization/output/ssp_trajectory_localization/May13_16-00-27/ssp_trajectory_localization_model.pt'
    else:
        model = '../output/ssp_path_integration/clipped/Mar22_15-24-10/ssp_path_integration_model.pt'
        model = "../output/ssp_path_integration/ssp_encoding_scaled_loss/gpu3runs/May14_14-31-33/ssp_path_integration_model.pt"
else:
    model = args.model

if args.use_localization:
    # fname_pred = 'loc_sac_{}samples_pred.pdf'.format(args.n_samples)
    # fname_truth = 'loc_sac_{}samples_truth.pdf'.format(args.n_samples)
    # This version has distance sensor measurements as well
    activations, predictions, coords = run_and_gather_localization_activations(
        n_samples=args.n_samples,
        dataset=dataset,
        model_path=model,
        encoding=args.encoding,
    )
else:
    # "../../lab/reproducing/data/path_integration_trajectories_logits_1000t_15s_seed13.npz"
    # ../output/ssp_path_integration/ssp_encoding_scaled_loss/gpu3runs/May14_14-31-33/ssp_path_integration_model.pt
    # fname_pred = 'sac_{}samples_pred.pdf'.format(args.n_samples)
    # fname_truth = 'sac_{}samples_truth.pdf'.format(args.n_samples)
    # fname_pred = 'scaled_hybrid_sac_{}samples_pred.pdf'.format(args.n_samples)
    # fname_truth = 'scaled_hybrid_sac_{}samples_truth.pdf'.format(args.n_samples)
    activations, predictions, coords = run_and_gather_activations(
        n_samples=args.n_samples,
        dataset=dataset,
        model_path=model,
        encoding=args.encoding,
        encoding_func=encoding_func,
    )

    predictions = predictions / ssp_scaling
    coords = coords / ssp_scaling

print(np.max(predictions))
print(np.min(predictions))
# assert False


# grid_scores['btln_60'], grid_scores['btln_90'], \
# grid_scores['btln_60_separation'], grid_scores['btln_90_separation'] = utils.get_scores_and_plot(
grid_scores_60_pred, grid_scores_90_pred, grid_scores_60_separation_pred, grid_scores_90_separation_pred = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=predictions, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_scores', #FLAGS.saver_results_directory,
    filename=fname_pred,
)

grid_scores_60_truth, grid_scores_90_truth, grid_scores_60_separation_truth, grid_scores_90_separation_truth = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=coords, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_scores', #FLAGS.saver_results_directory,
    filename=fname_truth,
)


print(grid_scores_60_truth, grid_scores_90_truth, grid_scores_60_separation_truth, grid_scores_90_separation_truth)

# Saving to make grid score values easy to compare for different variations
fname = '{}_{}samples.npz'.format(args.fname_prefix, args.n_samples)
np.savez(
    fname,
    grid_scores_60_pred=grid_scores_60_pred,
    grid_scores_90_pred=grid_scores_90_pred,
    grid_scores_60_separation_pred=grid_scores_60_separation_pred,
    grid_scores_90_separation_pred=grid_scores_90_separation_pred,
    grid_scores_60_truth=grid_scores_60_truth,
    grid_scores_90_truth=grid_scores_90_truth,
    grid_scores_60_separation_truth=grid_scores_60_separation_truth,
    grid_scores_90_separation_truth=grid_scores_90_separation_truth,
)
