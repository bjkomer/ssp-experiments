import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import numpy as np
import torch
# softlinked for now
from datasets import train_test_loaders
# softlinked for now
from models import SSPPathIntegrationModel
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
# softlinked for now
from path_integration_utils import pc_to_loc_v

# softlinked for now
from localization_training_utils import localization_train_test_loaders, LocalizationModel


def run_and_gather_activations(
        seed=13,
        n_samples=1000,
        dataset='../../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz',
        model_path='../output/ssp_path_integration/clipped/Mar22_15-24-10/ssp_path_integration_model.pt',
        encoding='ssp',
        rollout_length=100,
        batch_size=10,
        n_place_cells=256,

):

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(dataset)

    x_axis_vec = data['x_axis_vec']
    y_axis_vec = data['y_axis_vec']

    pc_centers = data['pc_centers']
    #pc_activations = data['pc_activations']

    if encoding == 'ssp':
        encoding_dim = 512
        ssp_scaling = data['ssp_scaling']
    elif encoding == '2d':
        encoding_dim = 2
        ssp_scaling = 1
    elif encoding == 'pc':
        dim = n_place_cells
        ssp_scaling = 1
    else:
        raise NotImplementedError

    limit_low = 0 * ssp_scaling
    limit_high = 2.2 * ssp_scaling
    res = 128 #256

    xs = np.linspace(limit_low, limit_high, res)
    ys = np.linspace(limit_low, limit_high, res)

    # Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

    model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=encoding_dim)

    model.load_state_dict(torch.load(model_path), strict=False)

    trainloader, testloader = train_test_loaders(
        data,
        n_train_samples=n_samples,
        n_test_samples=n_samples,
        rollout_length=rollout_length,
        batch_size=batch_size,
        encoding=encoding,
    )

    print("Testing")
    with torch.no_grad():
        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):
            velocity_inputs, ssp_inputs, ssp_outputs = data

            ssp_pred, lstm_outputs = model.forward_activations(velocity_inputs, ssp_inputs)


        predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
        coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
        activations = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], model.lstm_hidden_size))

        assert rollout_length == ssp_pred.shape[0]

        # # For each neuron, contains the average activity at each spatial bin
        # # Computing for both ground truth and predicted location
        # rate_maps_pred = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))
        # rate_maps_truth = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))

        print("Computing predicted locations and true locations")
        # Using all data, one chunk at a time
        for ri in range(rollout_length):

            if encoding == 'ssp':
                # computing 'predicted' coordinates, where the agent thinks it is
                predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[ri, :, :],
                    heatmap_vectors, xs, ys
                )

                # computing 'ground truth' coordinates, where the agent should be
                coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, ri, :],
                    heatmap_vectors, xs, ys
                )
            elif encoding == '2d':
                # copying 'predicted' coordinates, where the agent thinks it is
                predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_pred.detach().numpy()[ri, :, :]

                # copying 'ground truth' coordinates, where the agent should be
                coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_outputs.detach().numpy()[:, ri, :]
            elif encoding == 'pc':
                # (quick hack is to just use the most activated place cell center)
                predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = pc_to_loc_v(
                    pc_activations=ssp_outputs.detach().numpy()[:, ri, :],
                    centers=pc_centers,
                    jitter=0.01,
                )

                coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = pc_to_loc_v(
                    pc_activations=ssp_outputs.detach().numpy()[:, ri, :],
                    centers=pc_centers,
                    jitter=0.01,
                )

            # reshaping activations and converting to numpy array
            activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]

    return activations, predictions, coords


def run_and_gather_localization_activations(
        seed=13,
        n_samples=1000,
        dataset='../../localization/data/localization_trajectories_5m_200t_250s_seed13.npz',
        model_path='../../localization/output/ssp_trajectory_localization/May13_16-00-27/ssp_trajectory_localization_model.pt',
        encoding='ssp',
        rollout_length=100,
        batch_size=10,

):

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(dataset)

    x_axis_vec = data['x_axis_vec']
    y_axis_vec = data['y_axis_vec']
    ssp_scaling = data['ssp_scaling']
    ssp_offset = data['ssp_offset']

    # shape of coarse maps is (n_maps, env_size, env_size)
    # some npz files had different naming, try both
    try:
        coarse_maps = data['coarse_maps']
    except KeyError:
        coarse_maps = data['coarse_mazes']
    n_maps = coarse_maps.shape[0]
    env_size = coarse_maps.shape[1]

    # shape of dist_sensors is (n_maps, n_trajectories, n_steps, n_sensors)
    n_sensors = data['dist_sensors'].shape[3]

    # shape of ssps is (n_maps, n_trajectories, n_steps, dim)
    dim = data['ssps'].shape[3]

    limit_low = -ssp_offset * ssp_scaling
    limit_high = (env_size - ssp_offset) * ssp_scaling
    res = 256

    xs = np.linspace(limit_low, limit_high, res)
    ys = np.linspace(limit_low, limit_high, res)

    # Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

    model = LocalizationModel(
        input_size=2 + n_sensors + n_maps,
        unroll_length=rollout_length,
        sp_dim=dim
    )

    model.load_state_dict(torch.load(model_path), strict=False)

    trainloader, testloader = localization_train_test_loaders(
        data,
        n_train_samples=n_samples,
        n_test_samples=n_samples,
        rollout_length=rollout_length,
        batch_size=batch_size,
        encoding=encoding,
    )

    print("Testing")
    with torch.no_grad():
        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):
            combined_inputs, ssp_inputs, ssp_outputs = data

            ssp_pred, lstm_outputs = model.forward_activations(combined_inputs, ssp_inputs)

        predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
        coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
        activations = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], model.lstm_hidden_size))

        assert rollout_length == ssp_pred.shape[0]

        # # For each neuron, contains the average activity at each spatial bin
        # # Computing for both ground truth and predicted location
        # rate_maps_pred = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))
        # rate_maps_truth = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))

        print("Computing predicted locations and true locations")
        # Using all data, one chunk at a time
        for ri in range(rollout_length):

            if encoding == 'ssp':
                # computing 'predicted' coordinates, where the agent thinks it is
                predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[ri, :, :],
                    heatmap_vectors, xs, ys
                )

                # computing 'ground truth' coordinates, where the agent should be
                coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, ri, :],
                    heatmap_vectors, xs, ys
                )
            elif encoding == '2d':
                # copying 'predicted' coordinates, where the agent thinks it is
                predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_pred.detach().numpy()[ri, :, :]

                # copying 'ground truth' coordinates, where the agent should be
                coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_outputs.detach().numpy()[:, ri, :]

            # reshaping activations and converting to numpy array
            activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]

    return activations, predictions, coords
