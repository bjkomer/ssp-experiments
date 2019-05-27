import numpy as np
from spatial_semantic_pointers.utils import encode_point, encode_random, ssp_to_loc, ssp_to_loc_v, get_heatmap_vectors
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt


class TrajectoryValidationSet(object):

    def __init__(self, dataloader, heatmap_vectors, xs, ys, ssp_scaling=1, spatial_encoding='ssp'):

        self.dataloader = dataloader
        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys
        self.ssp_scaling = ssp_scaling
        self.spatial_encoding = spatial_encoding
        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.mse_criterion = nn.MSELoss()

    def run_eval(self, model, writer, epoch):

        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(self.dataloader):
                combined_inputs, ssp_inputs, ssp_outputs = data

                ssp_pred = model(combined_inputs, ssp_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                # NOTE: for cosine loss the input needs to be flattened first
                cosine_loss = self.cosine_criterion(
                    ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
                )
                mse_loss = self.mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

                print("test mse loss", mse_loss.data.item())
                print("test cosine loss", mse_loss.data.item())

            writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
            writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)

            # Just use start and end location to save on memory and computation
            predictions_start = np.zeros((ssp_pred.shape[1], 2))
            coords_start = np.zeros((ssp_pred.shape[1], 2))

            predictions_end = np.zeros((ssp_pred.shape[1], 2))
            coords_end = np.zeros((ssp_pred.shape[1], 2))

            if self.spatial_encoding == 'ssp':
                print("computing prediction locations")
                predictions_start[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[0, :, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                predictions_end[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[-1, :, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                print("computing ground truth locations")
                coords_start[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, 0, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                coords_end[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, -1, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
            elif self.spatial_encoding == '2d':
                print("copying prediction locations")
                predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
                predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
                print("copying ground truth locations")
                coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
                coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]

            fig_pred_start, ax_pred_start = plt.subplots()
            fig_truth_start, ax_truth_start = plt.subplots()
            fig_pred_end, ax_pred_end = plt.subplots()
            fig_truth_end, ax_truth_end = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(
                predictions_start / self.ssp_scaling,
                coords_start / self.ssp_scaling,
                ax_pred_start,
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )
            plot_predictions_v(
                predictions_end / self.ssp_scaling,
                coords_end / self.ssp_scaling,
                ax_pred_end,
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )

            writer.add_figure("predictions start", fig_pred_start, epoch)
            writer.add_figure("predictions end", fig_pred_end, epoch)

            # Only plotting ground truth if the epoch is 0
            if epoch == 0:

                print("plotting ground truth locations")
                plot_predictions_v(
                    coords_start / self.ssp_scaling,
                    coords_start / self.ssp_scaling,
                    ax_truth_start,
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )
                plot_predictions_v(
                    coords_end / self.ssp_scaling,
                    coords_end / self.ssp_scaling,
                    ax_truth_end,
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )

                writer.add_figure("ground truth start", fig_truth_start, epoch)
                writer.add_figure("ground truth end", fig_truth_end, epoch)


class SnapshotValidationSet(object):

    def __init__(self, dataloader, heatmap_vectors, xs, ys, spatial_encoding='ssp'):

        self.dataloader = dataloader
        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys
        self.spatial_encoding = spatial_encoding
        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.mse_criterion = nn.MSELoss()

    def run_eval(self, model, writer, epoch):

        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(self.dataloader):
                # sensor_inputs, map_ids, ssp_outputs = data
                # sensors and map ID combined
                combined_inputs, ssp_outputs = data

                # ssp_pred = model(sensor_inputs, map_ids)
                ssp_pred = model(combined_inputs)

                cosine_loss = self.cosine_criterion(ssp_pred, ssp_outputs, torch.ones(ssp_pred.shape[0]))
                mse_loss = self.mse_criterion(ssp_pred, ssp_outputs)

                print("test mse loss", mse_loss.data.item())
                print("test cosine loss", mse_loss.data.item())

            writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
            writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)

            # One prediction and ground truth coord for every element in the batch
            # NOTE: this is assuming the eval set only has one giant batch
            predictions = np.zeros((ssp_pred.shape[0], 2))
            coords = np.zeros((ssp_pred.shape[0], 2))

            if self.spatial_encoding == 'ssp':
                print("computing prediction locations")
                predictions[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[:, :],
                    self.heatmap_vectors, self.xs, self.ys
                )

                print("computing ground truth locations")
                coords[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, :],
                    self.heatmap_vectors, self.xs, self.ys
                )

            elif self.spatial_encoding == '2d':
                print("copying prediction locations")
                predictions[:, :] = ssp_pred.detach().numpy()[:, :]
                print("copying ground truth locations")
                coords[:, :] = ssp_outputs.detach().numpy()[:, :]

            fig_pred, ax_pred = plt.subplots()
            fig_truth, ax_truth = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(
                # predictions / self.ssp_scaling,
                # coords / self.ssp_scaling,
                predictions,
                coords,
                ax_pred,
                # min_val=0,
                # max_val=2.2
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )

            writer.add_figure("predictions", fig_pred, epoch)

            # Only plot ground truth if epoch is 0
            if epoch == 0:
                print("plotting ground truth locations")
                plot_predictions_v(
                    # coords / self.ssp_scaling,
                    # coords / self.ssp_scaling,
                    coords,
                    coords,
                    ax_truth,
                    # min_val=0,
                    # max_val=2.2
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )

                writer.add_figure("ground truth", fig_truth, epoch)


class LocalizationTrajectoryDataset(data.Dataset):

    def __init__(self, velocity_inputs, sensor_inputs, maze_ids, ssp_inputs, ssp_outputs, return_velocity_list=True):

        self.velocity_inputs = velocity_inputs.astype(np.float32)
        self.sensor_inputs = sensor_inputs.astype(np.float32)
        self.maze_ids = maze_ids.astype(np.float32)

        # self.combined_inputs = np.hstack([self.velocity_inputs, self.sensor_inputs, self.maze_ids])
        self.combined_inputs = np.concatenate([self.velocity_inputs, self.sensor_inputs, self.maze_ids], axis=2)
        assert (self.velocity_inputs.shape[0] == self.combined_inputs.shape[0])
        assert (self.velocity_inputs.shape[1] == self.combined_inputs.shape[1])
        assert (self.combined_inputs.shape[2] == self.velocity_inputs.shape[2] + self.sensor_inputs.shape[2] + self.maze_ids.shape[2])
        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.ssp_outputs = ssp_outputs.astype(np.float32)

        # flag for whether the velocities returned are a single tensor or a list of tensors
        self.return_velocity_list = return_velocity_list

    def __getitem__(self, index):

        if self.return_velocity_list:
            return [self.combined_inputs[index, i] for i in range(self.combined_inputs.shape[1])], \
                   self.ssp_inputs[index], self.ssp_outputs[index],
        else:
            return self.combined_inputs[index], self.ssp_inputs[index], self.ssp_outputs[index]

    def __len__(self):
        return self.combined_inputs.shape[0]


class LocalizationSnapshotDataset(data.Dataset):

    def __init__(self, sensor_inputs, maze_ids, ssp_outputs):

        self.sensor_inputs = sensor_inputs.astype(np.float32)
        self.maze_ids = maze_ids.astype(np.float32)
        self.ssp_outputs = ssp_outputs.astype(np.float32)
        self.combined_inputs = np.hstack([self.sensor_inputs, self.maze_ids])
        assert(self.sensor_inputs.shape[0] == self.combined_inputs.shape[0])

    def __getitem__(self, index):

        # return self.sensor_inputs[index], self.maze_ids[index], self.ssp_outputs[index]
        return self.combined_inputs[index], self.ssp_outputs[index]

    def __len__(self):
        return self.sensor_inputs.shape[0]


def localization_train_test_loaders(
        data, n_train_samples=1000, n_test_samples=1000,
        rollout_length=100, batch_size=10, encoding='ssp', n_mazes_to_use=0
):

    # Option to use SSPs or the 2D location directly
    assert encoding in ['ssp', '2d', 'pc']

    positions = data['positions']

    dist_sensors = data['dist_sensors']
    n_sensors = dist_sensors.shape[3]

    cartesian_vels = data['cartesian_vels']
    ssps = data['ssps']
    # n_place_cells = data['pc_centers'].shape[0]
    #
    # pc_activations = data['pc_activations']

    coarse_maps = data['coarse_maps']
    n_mazes = coarse_maps.shape[0]

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]
    dim = ssps.shape[3]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        sensor_inputs = np.zeros((n_samples, rollout_length, n_sensors))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, dim))

        ssp_inputs = np.zeros((n_samples, dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        # # for the place cell encoding method
        # pc_outputs = np.zeros((n_samples, rollout_length, n_place_cells))
        #
        # pc_inputs = np.zeros((n_samples, n_place_cells))

        maze_ids = np.zeros((n_samples, rollout_length, n_mazes))

        for i in range(n_samples):
            # choose random map
            if n_mazes_to_use <= 0:
                # use all available mazes
                maze_ind = np.random.randint(low=0, high=n_mazes)
            else:
                # use only some mazes
                maze_ind = np.random.randint(low=0, high=n_mazes_to_use)
            # choose random trajectory
            traj_ind = np.random.randint(low=0, high=n_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, :] = cartesian_vels[maze_ind, traj_ind, step_ind:step_ind_final, :]

            sensor_inputs[i, :, :] = dist_sensors[maze_ind, traj_ind, step_ind:step_ind_final, :]

            # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
            ssp_outputs[i, :, :] = ssps[maze_ind, traj_ind, step_ind + 1:step_ind_final + 1, :]
            # initial state of the LSTM is a linear transform of the ground truth ssp
            ssp_inputs[i, :] = ssps[maze_ind, traj_ind, step_ind]

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[maze_ind, traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[maze_ind, traj_ind, step_ind]

            # # for the place cell encoding method
            # pc_outputs[i, :, :] = pc_activations[traj_ind, step_ind + 1:step_ind_final + 1, :]
            # pc_inputs[i, :] = pc_activations[traj_ind, step_ind]

            # one-hot maze ID
            maze_ids[i, :, maze_ind] = 1

        if encoding == 'ssp':
            dataset = LocalizationTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
            )
        elif encoding == '2d':
            dataset = LocalizationTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_inputs=pos_inputs,
                ssp_outputs=pos_outputs,
            )
        # elif encoding == 'pc':
        #     dataset = LocalizationTrajectoryDataset(
        #         velocity_inputs=velocity_inputs,
        #         sensor_inputs=sensor_inputs,
        #         ssp_inputs=pc_inputs,
        #         ssp_outputs=pc_outputs,
        #     )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0,
            )

    return trainloader, testloader


# TODO: need to handle multiple mazes still
def snapshot_localization_train_test_loaders(
        data, n_train_samples=1000, n_test_samples=1000, batch_size=10, encoding='ssp', n_mazes_to_use=0
):

    # Option to use SSPs or the 2D location directly
    assert encoding in ['ssp', '2d']

    xs = data['xs']
    ys = data['ys']
    x_axis_vec = data['x_axis_sp']
    y_axis_vec = data['y_axis_sp']
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

    # positions = data['positions']

    # shape is (n_mazes, res, res, n_sensors)
    dist_sensors = data['dist_sensors']

    fine_mazes = data['fine_mazes']

    n_sensors = dist_sensors.shape[3]

    # ssps = data['ssps']

    n_mazes = data['coarse_mazes'].shape[0]
    dim = x_axis_vec.shape[0]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        sensor_inputs = np.zeros((n_samples, n_sensors))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, 2))

        maze_ids = np.zeros((n_samples, n_mazes))

        for i in range(n_samples):
            # choose random maze and position in maze
            if n_mazes_to_use <= 0:
                # use all available mazes
                maze_ind = np.random.randint(low=0, high=n_mazes)
            else:
                # use only some mazes
                maze_ind = np.random.randint(low=0, high=n_mazes_to_use)
            xi = np.random.randint(low=0, high=len(xs))
            yi = np.random.randint(low=0, high=len(ys))
            # Keep choosing position until it is not inside a wall
            while fine_mazes[maze_ind, xi, yi] == 1:
                xi = np.random.randint(low=0, high=len(xs))
                yi = np.random.randint(low=0, high=len(ys))

            sensor_inputs[i, :] = dist_sensors[maze_ind, xi, yi, :]

            ssp_outputs[i, :] = heatmap_vectors[xi, yi, :]

            # one-hot maze ID
            maze_ids[i, maze_ind] = 1

            # for the 2D encoding method
            pos_outputs[i, :] = np.array([xs[xi], ys[yi]])

        if encoding == 'ssp':
            dataset = LocalizationSnapshotDataset(
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_outputs=ssp_outputs,
            )
        elif encoding == '2d':
            dataset = LocalizationSnapshotDataset(
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_outputs=pos_outputs,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0,
            )

    return trainloader, testloader


class LocalizationModel(nn.Module):

    def __init__(self, input_size, lstm_hidden_size=128, linear_hidden_size=512,
                 unroll_length=100, sp_dim=512, dropout_p=0.5):

        super(LocalizationModel, self).__init__()

        self.input_size = input_size  # velocity and sensor measurements
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.unroll_length = unroll_length

        self.sp_dim = sp_dim

        # Full LSTM that can be given the full sequence and produce the full output in one step
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1
        )

        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=self.linear_hidden_size,
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.ssp_output = nn.Linear(
            in_features=self.linear_hidden_size,
            out_features=self.sp_dim
        )

        # Linear transforms for ground truth ssp into initial hidden and cell state of lstm
        self.w_c = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_h = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

    def forward(self, inputs, initial_ssp):
        """
        :param inputs: contains both velocity and distance sensor measurments (and potentially map_id)
        :param initial_ssp: SSP ground truth for the start location of the trajectory
        :return: predicted SSP for agent location at the end of the trajectory
        """

        ssp_pred, output = self.forward_activations(inputs, initial_ssp)
        return ssp_pred

    def forward_activations(self, inputs, initial_ssp):
        """Returns the hidden layer activations as well as the prediction"""

        batch_size = inputs[0].shape[0]

        # Compute initial hidden state
        cell_state = self.w_c(initial_ssp)
        hidden_state = self.w_h(initial_ssp)

        vel_sense_inputs = torch.cat(inputs).view(len(inputs), batch_size, -1)

        output, (_, _) = self.lstm(
            vel_sense_inputs,
            (
                hidden_state.view(1, batch_size, self.lstm_hidden_size),
                cell_state.view(1, batch_size, self.lstm_hidden_size)
            )
        )

        features = self.dropout(self.linear(output))

        # TODO: should normalization be used here?
        ssp_pred = self.ssp_output(features)

        return ssp_pred, output


# TODO: have this somewhere globally accessible instead of many copies
class FeedForward(nn.Module):
    """
    Single hidden layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction, features


def pc_to_loc_v(pc_activations, centers, jitter=0.01):
    """
    Approximate decoding of place cell activations.
    Rounding to the nearest place cell center. Just to get a sense of whether the output is in the right ballpark
    :param pc_activations: activations of each place cell, of shape (n_samples, n_place_cells)
    :param centers: centers of each place cell, of shape (n_place_cells, 2)
    :param jitter: noise to add to the output, so locations on top of each other can be seen
    :return: array of the 2D coordinates that the place cell activation most closely represents
    """

    n_samples = pc_activations.shape[0]

    indices = np.argmax(pc_activations, axis=1)

    return centers[indices] + np.random.normal(loc=0, scale=jitter, size=(n_samples, 2))
