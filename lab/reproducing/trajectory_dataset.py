import numpy as np
import torch.utils.data as data
import torch


class TrajectoryDataset(data.Dataset):

    def __init__(self, velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs, return_velocity_list=True):

        self.velocity_inputs = velocity_inputs.astype(np.float32)
        self.pc_inputs = pc_inputs.astype(np.float32)
        self.hd_inputs = hd_inputs.astype(np.float32)
        self.pc_outputs = pc_outputs.astype(np.float32)
        self.hd_outputs = hd_outputs.astype(np.float32)

        # flag for whether the velocities returned are a single tensor or a list of tensors
        self.return_velocity_list = return_velocity_list

    def __getitem__(self, index):

        if self.return_velocity_list:
            return [self.velocity_inputs[index, i] for i in range(self.velocity_inputs.shape[1])],\
                   self.pc_inputs[index], self.hd_inputs[index], self.pc_outputs[index], self.hd_outputs[index]
        else:
            return self.velocity_inputs[index], self.pc_inputs[index], self.hd_inputs[index], self.pc_outputs[index], \
                   self.hd_outputs[index]

    def __len__(self):
        return self.velocity_inputs.shape[0]


def train_test_loaders(data, n_train_samples=1000, n_test_samples=1000, rollout_length=100, batch_size=10):
    positions = data['positions']
    angles = data['angles']
    lin_vels = data['lin_vels']
    ang_vels = data['ang_vels']
    cos_ang_vels = np.cos(ang_vels)
    sin_ang_vels = np.sin(ang_vels)
    pc_activations = data['pc_activations']
    hd_activations = data['hd_activations']

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]
    n_place_cells = pc_activations.shape[2]
    n_hd_cells = hd_activations.shape[2]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 3))
        activation_outputs = np.zeros((n_samples, n_place_cells + n_hd_cells))
        pc_outputs = np.zeros((n_samples, n_place_cells))
        hd_outputs = np.zeros((n_samples, n_hd_cells))

        # these include outputs for every time-step
        full_pc_outputs = np.zeros((n_samples, rollout_length, n_place_cells))
        full_hd_outputs = np.zeros((n_samples, rollout_length, n_hd_cells))

        pc_inputs = np.zeros((n_samples, n_place_cells))
        hd_inputs = np.zeros((n_samples, n_hd_cells))

        for i in range(n_samples):
            # choose random trajectory
            traj_ind = np.random.randint(low=0, high=n_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length - 1

            velocity_inputs[i, :, 0] = lin_vels[traj_ind, step_ind:step_ind_final + 1]
            velocity_inputs[i, :, 1] = cos_ang_vels[traj_ind, step_ind:step_ind_final + 1]
            velocity_inputs[i, :, 2] = sin_ang_vels[traj_ind, step_ind:step_ind_final + 1]

            activation_outputs[i, :n_place_cells] = pc_activations[traj_ind, step_ind_final]
            activation_outputs[i, n_place_cells:] = hd_activations[traj_ind, step_ind_final]
            pc_outputs[i, :] = pc_activations[traj_ind, step_ind_final]
            hd_outputs[i, :] = hd_activations[traj_ind, step_ind_final]

            full_pc_outputs[i, :] = pc_activations[traj_ind, step_ind:step_ind_final + 1]
            full_hd_outputs[i, :] = hd_activations[traj_ind, step_ind:step_ind_final + 1]

            # initial state of the LSTM is a linear transform of the ground truth place and hd cell activations
            pc_inputs[i, :] = pc_activations[traj_ind, step_ind]
            hd_inputs[i, :] = hd_activations[traj_ind, step_ind]

        dataset = TrajectoryDataset(
            velocity_inputs=velocity_inputs,
            pc_inputs=pc_inputs,
            hd_inputs=hd_inputs,

            # # TODO: fix this to have the whole trajectory
            # pc_outputs=pc_outputs,
            # hd_outputs=hd_outputs,

            # TODO: fix this to have the whole trajectory
            pc_outputs=full_pc_outputs,
            hd_outputs=full_hd_outputs,
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
