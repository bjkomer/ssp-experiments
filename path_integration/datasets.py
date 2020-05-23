import numpy as np
import torch.utils.data as data
import torch


class SSPTrajectoryDataset(data.Dataset):

    def __init__(self, velocity_inputs, ssp_inputs, ssp_outputs, return_velocity_list=True, device=torch.device('cpu:0')):

        self.device = device

        self.velocity_inputs = torch.Tensor(velocity_inputs.astype(np.float32)).to(self.device)
        self.ssp_inputs = torch.Tensor(ssp_inputs.astype(np.float32)).to(self.device)
        self.ssp_outputs = torch.Tensor(ssp_outputs.astype(np.float32)).to(self.device)

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
                       batch_size=10, encoding='ssp', encoding_func=None, encoding_dim=512,
                       hd_encoding_func=None, hd_dim=0,
                       train_split=0.8,
                       pin_memory=False,
                       device='cpu',
                       ):
    # Option to use SSPs or the 2D location directly
    # assert encoding in ['ssp', '2d', 'pc', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq']

    positions = data['positions']

    cartesian_vels = data['cartesian_vels']

    print("Generating encodings for every point")
    if encoding != '2d':
        if encoding_func is None and encoding == 'ssp':
            # backwards compatibility with older datasets
            ssps = data['ssps']
        else:
            assert encoding_func is not None

            ssps = np.zeros((positions.shape[0], positions.shape[1], encoding_dim))

            for traj in range(ssps.shape[0]):
                for step in range(ssps.shape[1]):
                    ssps[traj, step, :] = encoding_func(
                        # positions=positions[traj, step, :]
                        x=positions[traj, step, 0],
                        y=positions[traj, step, 1]
                    )
    print("Encoding Generation Complete")

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]

    # Split training and testing to be independent
    n_train_trajectories = int(train_split * n_trajectories)

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, encoding_dim))

        ssp_inputs = np.zeros((n_samples, encoding_dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # choose random trajectory
            if test_set == 0:
                traj_ind = np.random.randint(low=n_train_trajectories, high=n_trajectories)
            else:
                traj_ind = np.random.randint(low=0, high=n_train_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, :] = cartesian_vels[traj_ind, step_ind:step_ind_final, :]

            if encoding != '2d':
                # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
                ssp_outputs[i, :, :] = ssps[traj_ind, step_ind + 1:step_ind_final + 1, :]
                # initial state of the LSTM is a linear transform of the ground truth ssp
                ssp_inputs[i, :] = ssps[traj_ind, step_ind]

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[traj_ind, step_ind]

        if encoding == '2d':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=pos_inputs,
                ssp_outputs=pos_outputs,
                device=device,
            )
        else:
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
                device=device,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0, pin_memory=pin_memory
            )

    return trainloader, testloader


def train_test_loaders_jit(data, n_train_samples=1000, n_test_samples=1000, rollout_length=100,
                       batch_size=10, encoding='ssp', encoding_func=None, encoding_dim=512,
                       hd_encoding_func=None, hd_dim=0,
                       train_split=0.8,
                       pin_memory=False,
                       device='cpu',
                       rng=np.random
                       ):
    # this version doesn't precompute all encodings, just encodes as needed

    positions = data['positions']

    cartesian_vels = data['cartesian_vels']

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]

    # Split training and testing to be independent
    n_train_trajectories = int(train_split * n_trajectories)

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, encoding_dim))

        ssp_inputs = np.zeros((n_samples, encoding_dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # choose random trajectory
            if test_set == 0:
                traj_ind = rng.randint(low=n_train_trajectories, high=n_trajectories)
            else:
                traj_ind = rng.randint(low=0, high=n_train_trajectories)
            # choose random segment of trajectory
            step_ind = rng.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, :] = cartesian_vels[traj_ind, step_ind:step_ind_final, :]

            if encoding != '2d':
                # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
                # ssp_outputs[i, :, :] = ssps[traj_ind, step_ind + 1:step_ind_final + 1, :]
                for ji, j in enumerate(range(step_ind + 1, step_ind_final + 1)):
                    ssp_outputs[i, ji, :] = encoding_func(
                        x=positions[traj_ind, j, 0],
                        y=positions[traj_ind, j, 1]
                    )
                # initial state of the LSTM is a linear transform of the ground truth ssp
                # ssp_inputs[i, :] = ssps[traj_ind, step_ind]
                ssp_inputs[i, :] = encoding_func(
                    x=positions[traj_ind, step_ind, 0],
                    y=positions[traj_ind, step_ind, 1]
                )

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[traj_ind, step_ind]

        if encoding == '2d':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=pos_inputs,
                ssp_outputs=pos_outputs,
                device=device,
            )
        else:
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
                device=device,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0, pin_memory=pin_memory
            )

    return trainloader, testloader

def angular_train_test_loaders(data, n_train_samples=1000, n_test_samples=1000, rollout_length=100,
                       batch_size=10, encoding='ssp', encoding_func=None, encoding_dim=512,
                       hd_encoding_func=None, hd_dim=0,
                       train_split=0.8,
                       sin_cos_ang=True,
                       pin_memory=False,
                       device='cpu',
                       ):
    # Option to use SSPs or the 2D location directly
    # assert encoding in ['ssp', '2d', 'pc', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq']

    # This function should only be called when hd_dim is used
    assert hd_dim > 0
    assert hd_encoding_func is not None

    positions = data['positions']

    cartesian_vels = data['cartesian_vels']

    angles = data['angles']
    lin_vels = data['lin_vels']
    ang_vels = data['ang_vels']

    print("Generating encodings for every point")

    ssps = np.zeros((positions.shape[0], positions.shape[1], encoding_dim))
    hd_rep = np.zeros((angles.shape[0], angles.shape[1], hd_dim))

    for traj in range(ssps.shape[0]):
        for step in range(ssps.shape[1]):
            if encoding != '2d':
                ssps[traj, step, :] = encoding_func(
                    # positions=positions[traj, step, :]
                    x=positions[traj, step, 0],
                    y=positions[traj, step, 1],
                )
            hd_rep[traj, step, :] = hd_encoding_func(
                angle=angles[traj, step]
            )

    print("Encoding Generation Complete")

    n_trajectories = positions.shape[0]
    trajectory_length = positions.shape[1]

    # Split training and testing to be independent
    n_train_trajectories = int(train_split * n_trajectories)

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        if sin_cos_ang:
            # uses the sin and cos of the angular velocity, to match the deepmind paper
            velocity_inputs = np.zeros((n_samples, rollout_length, 3))
        else:
            velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, encoding_dim))

        ssp_inputs = np.zeros((n_samples, encoding_dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        ang_outputs = np.zeros((n_samples, rollout_length))
        ang_inputs = np.zeros((n_samples))

        hd_rep_outputs = np.zeros((n_samples, rollout_length, hd_dim))
        hd_rep_inputs = np.zeros((n_samples, hd_dim))

        for i in range(n_samples):
            # choose random trajectory
            if test_set == 0:
                traj_ind = np.random.randint(low=n_train_trajectories, high=n_trajectories)
            else:
                traj_ind = np.random.randint(low=0, high=n_train_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, 0] = lin_vels[traj_ind, step_ind:step_ind_final]
            if sin_cos_ang:
                velocity_inputs[i, :, 1] = np.sin(ang_vels[traj_ind, step_ind:step_ind_final])
                velocity_inputs[i, :, 2] = np.cos(ang_vels[traj_ind, step_ind:step_ind_final])
            else:
                velocity_inputs[i, :, 1] = ang_vels[traj_ind, step_ind:step_ind_final]

            # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
            ssp_outputs[i, :, :] = ssps[traj_ind, step_ind + 1:step_ind_final + 1, :]
            hd_rep_outputs[i, :, :] = hd_rep[traj_ind, step_ind + 1:step_ind_final + 1, :]
            # initial state of the LSTM is a linear transform of the ground truth ssp
            ssp_inputs[i, :] = ssps[traj_ind, step_ind]
            hd_rep_inputs[i, :] = hd_rep[traj_ind, step_ind]

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[traj_ind, step_ind]

            ang_outputs[i, :] = angles[traj_ind, step_ind + 1:step_ind_final + 1]
            ang_inputs[i] = angles[traj_ind, step_ind]

        if encoding == '2d':
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=np.concatenate([pos_inputs, hd_rep_inputs], axis=1),
                ssp_outputs=np.concatenate([pos_outputs, hd_rep_outputs], axis=2),
                device=device,
            )
        else:
            dataset = SSPTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                ssp_inputs=np.concatenate([ssp_inputs, hd_rep_inputs], axis=1),
                ssp_outputs=np.concatenate([ssp_outputs, hd_rep_outputs], axis=2),
                device=device,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0, pin_memory=pin_memory
            )

    return trainloader, testloader


def tf_train_test_loaders(data, n_train_samples=500000, n_test_samples=500000, rollout_length=100,
                       batch_size=10, encoding='ssp', encoding_func=None, encoding_dim=512,
                       hd_encoding_func=None, hd_dim=0,
                       train_split=0.8,
                       sin_cos_ang=True,
                       ):

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        if test_set == 0:
            print("Loading data for training set")
            init_pos = data['init_pos'][:n_samples]
            init_hd = data['init_hd'][:n_samples]
            ego_vel = data['ego_vel'][:n_samples]
            target_pos = data['target_pos'][:n_samples]
            target_hd = data['target_hd'][:n_samples]
        else:
            print("Loading data for testing set")
            init_pos = data['init_pos'][n_samples:]
            init_hd = data['init_hd'][n_samples:]
            ego_vel = data['ego_vel'][n_samples:]
            target_pos = data['target_pos'][n_samples:]
            target_hd = data['target_hd'][n_samples:]

        ssp_inputs = np.zeros((init_pos.shape[0], encoding_dim + hd_dim))
        ssp_outputs = np.zeros((target_pos.shape[0], target_pos.shape[1], encoding_dim + hd_dim))

        print("Generating encodings for every point")

        # inputs
        for traj in range(ssp_inputs.shape[0]):

            ssp_inputs[traj, :encoding_dim] = encoding_func(
                x=init_pos[traj, 0],
                y=init_pos[traj, 1],
            )

            if hd_dim > 0:
                ssp_inputs[traj, encoding_dim:] = hd_encoding_func(
                    angle=init_hd[traj]
                )

        # outputs
        for traj in range(ssp_outputs.shape[0]):
            for step in range(ssp_outputs.shape[1]):
                # if encoding != '2d':
                #     ssp = encoding_func(
                #         # positions=positions[traj, step, :]
                #         x=positions[traj, step, 0],
                #         y=positions[traj, step, 1],
                #     )
                # hd_rep = hd_encoding_func(
                #     angle=angles[traj, step]
                # )

                ssp_outputs[traj, step, :encoding_dim] = encoding_func(
                        x=target_pos[traj, step, 0],
                        y=target_pos[traj, step, 1],
                    )

                if hd_dim > 0:
                    ssp_outputs[traj, step, encoding_dim:] = hd_encoding_func(
                        angle=target_hd[traj, step]
                    )

        print("Encoding Generation Complete")

        dataset = SSPTrajectoryDataset(
            velocity_inputs=ego_vel,
            ssp_inputs=ssp_inputs,
            ssp_outputs=ssp_outputs,
        )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            )
        else:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0,
            )

    return trainloader, testloader


def load_from_cache(fname, batch_size=10, n_samples=1000, device='cpu'):

    data = np.load(fname)

    dataset_train = SSPTrajectoryDataset(
        velocity_inputs=data['train_velocity_inputs'],
        ssp_inputs=data['train_ssp_inputs'],
        ssp_outputs=data['train_ssp_outputs'],
        device=device,
    )

    dataset_test = SSPTrajectoryDataset(
        velocity_inputs=data['test_velocity_inputs'],
        ssp_inputs=data['test_ssp_inputs'],
        ssp_outputs=data['test_ssp_outputs'],
        device=device,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=n_samples, shuffle=True, num_workers=0,
    )

    return trainloader, testloader

