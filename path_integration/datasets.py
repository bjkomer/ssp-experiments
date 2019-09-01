import numpy as np
import torch.utils.data as data
import torch


class SSPTrajectoryDataset(data.Dataset):

    def __init__(self, velocity_inputs, ssp_inputs, ssp_outputs, return_velocity_list=True):

        self.velocity_inputs = velocity_inputs.astype(np.float32)
        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.ssp_outputs = ssp_outputs.astype(np.float32)

        # flag for whether the velocities returned are a single tensor or a list of tensors
        self.return_velocity_list = return_velocity_list

    def __getitem__(self, index):

        if self.return_velocity_list:
            return [self.velocity_inputs[index, i] for i in range(self.velocity_inputs.shape[1])], \
                   self.ssp_inputs[index], self.ssp_outputs[index],
        else:
            return self.velocity_inputs[index], self.ssp_inputs[index], self.ssp_outputs[index]

    def __len__(self):
        return self.velocity_inputs.shape[0]


def train_test_loaders(data, n_train_samples=1000, n_test_samples=1000, rollout_length=100,
                       batch_size=10, encoding='ssp', encoding_func=None):

    # Option to use SSPs or the 2D location directly
    assert encoding in ['ssp', '2d', 'pc', 'frozen-learned']

    positions = data['positions']

    cartesian_vels = data['cartesian_vels']

    if encoding == 'frozen-learned':  # maybe just make this custom-encoding instead?
        assert encoding_func is not None
        # FIXME: make this less hacky and hardcoded
        ssps = np.zeros((positions.shape[0], positions.shape[1], 512))
        # for traj in range(ssps.shape[0]):
        #     for step in range(ssps.shape[1]):
        #         ssps[traj, step, :] = encoding_func(
        #             x=positions[traj, step, 0], y=positions[traj, step, 1]
        #         )
        for traj in range(ssps.shape[0]):
            for step in range(ssps.shape[1]):
                ssps[traj, :, :] = encoding_func(
                    positions=positions[traj, :, :]
                )

    else:
        ssps = data['ssps']
    n_place_cells = data['pc_centers'].shape[0]

    pc_activations = data['pc_activations']

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]
    dim = ssps.shape[2]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, dim))

        ssp_inputs = np.zeros((n_samples, dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        # for the place cell encoding method
        pc_outputs = np.zeros((n_samples, rollout_length, n_place_cells))

        pc_inputs = np.zeros((n_samples, n_place_cells))

        for i in range(n_samples):
            # choose random trajectory
            traj_ind = np.random.randint(low=0, high=n_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, :] = cartesian_vels[traj_ind, step_ind:step_ind_final, :]

            # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
            ssp_outputs[i, :, :] = ssps[traj_ind, step_ind + 1:step_ind_final + 1, :]
            # initial state of the LSTM is a linear transform of the ground truth ssp
            ssp_inputs[i, :] = ssps[traj_ind, step_ind]

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[traj_ind, step_ind]

            # for the place cell encoding method
            pc_outputs[i, :, :] = pc_activations[traj_ind, step_ind + 1:step_ind_final + 1, :]
            pc_inputs[i, :] = pc_activations[traj_ind, step_ind]

        if encoding == 'ssp':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
            )
        elif encoding == '2d':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=pos_inputs,
                ssp_outputs=pos_outputs,
            )
        elif encoding == 'pc':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=pc_inputs,
                ssp_outputs=pc_outputs,
            )
        elif encoding == 'frozen-learned':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
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
