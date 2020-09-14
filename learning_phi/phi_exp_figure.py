import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns

quick = False
# quick = True

# for losses
df = pd.DataFrame()

# for percent of params learned correctly
df_params = pd.DataFrame()

eps = 0.01


if quick:
    fname_base = 'output/quick_learned_phi_results_bs{}_{}D_{}dim_{}limit.npz'

    limits = [0.5, 1.0, 2.0, 4.0]
    ssp_dims = [8, 16, 32, 64, 128, 256]
    batch_sizes = [32, 256, 1024]
    coord_dims = [1, 2]

    for batch_size in batch_sizes:
        for limit in limits:
            for ssp_dim in ssp_dims:
                for coord_dim in coord_dims:

                    try:
                        data = np.load(fname_base.format(batch_size, coord_dim, ssp_dim, limit))

                        n_samples = data['losses'].shape[0]

                        for i in range(n_samples):
                            df = df.append(
                                {
                                    'Train Loss': data['losses'][i, -1],
                                    'Test Loss': data['val_losses'][i, -1],
                                    'Batch Size': batch_size,
                                    'Limit': limit,
                                    'SSP Dim': ssp_dim,
                                    'Coord Dim': coord_dim,
                                },
                                ignore_index=True
                            )

                            true_phis = data['true_phis'][i, :, :]
                            learned_phis = data['learned_phis'][i, :, :]

                            # convert learned phis to be between -pi and pi
                            while True:
                                ind_above = learned_phis > np.pi
                                if len(np.where(ind_above)[0]) > 0:
                                    learned_phis[ind_above] -= 2 * np.pi
                                ind_below = learned_phis < -np.pi
                                if len(np.where(ind_below)[0]) > 0:
                                    learned_phis[ind_below] += 2 * np.pi
                                if len(np.where(ind_above)[0]) == 0 and len(np.where(ind_below)[0]) == 0:
                                    break

                            matches = np.abs(learned_phis.flatten() - true_phis.flatten()) < eps
                            param_match = len(np.where(matches)[0]) / len(learned_phis.flatten())

                            param_rmse = np.sqrt(np.mean((learned_phis.flatten() - true_phis.flatten()) ** 2))

                            df_params = df_params.append(
                                {
                                    'Param Match': param_match,
                                    'Param RMSE': param_rmse,
                                    'Batch Size': batch_size,
                                    'Limit': limit,
                                    'SSP Dim': ssp_dim,
                                    'Coord Dim': coord_dim,
                                },
                                ignore_index=True
                            )
                    except:
                        print("File not found: {}".format(fname_base.format(batch_size, coord_dim, ssp_dim, limit)))
                        continue

    for limit in limits:
        try:
            df_tmp = df[df['Limit'] == limit]
            plt.figure()
            sns.barplot(data=df_tmp, x='SSP Dim', y='Test Loss', hue='Coord Dim')
            # sns.barplot(data=df_tmp, x='SSP Dim', y='Test Loss', hue='Batch Size')
            plt.title("Limit: {}".format(limit))

            df_tmp_params = df_params[df_params['Limit'] == limit]
            plt.figure()
            sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param Match', hue='Coord Dim')
            # sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param Match', hue='Batch Size')
            plt.title("Limit: {}".format(limit))

            plt.figure()
            sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param RMSE', hue='Coord Dim')
            # sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param RMSE', hue='Batch Size')
            plt.title("Limit: {}".format(limit))
        except:
            print("no data for limit: {}".format(limit))
            continue

else:

    fname_base = 'output/learned_phi_results_{}D_{}dim_{}limit.npz'

    limits = [0.5, 1.0, 2.0, 4.0]
    ssp_dims = [5, 10, 15, 20, 100]
    coord_dims = [1, 2]

    # fname_base = 'output/single_step_learned_phi_results_{}D_{}dim_{}limit.npz'
    # ssp_dims = [5, 20]

    for limit in limits:
        for ssp_dim in ssp_dims:
            for coord_dim in coord_dims:

                try:
                    data = np.load(fname_base.format(coord_dim, ssp_dim, limit))

                    n_samples = data['losses'].shape[0]

                    for i in range(n_samples):
                        df = df.append(
                            {
                                'Train Loss': data['losses'][i, -1],
                                'Test Loss': data['val_losses'][i, -1],
                                'Limit': limit,
                                'SSP Dim': ssp_dim,
                                'Coord Dim': coord_dim,
                            },
                            ignore_index=True
                        )

                        true_phis = data['true_phis'][i, :, :]
                        learned_phis = data['learned_phis'][i, :, :]

                        # convert learned phis to be between -pi and pi
                        while True:
                            ind_above = learned_phis > np.pi
                            if len(np.where(ind_above)[0]) > 0:
                                learned_phis[ind_above] -= 2*np.pi
                            ind_below = learned_phis < -np.pi
                            if len(np.where(ind_below)[0]) > 0:
                                learned_phis[ind_below] += 2 * np.pi
                            if len(np.where(ind_above)[0]) == 0 and len(np.where(ind_below)[0]) == 0:
                                break

                        matches = np.abs(learned_phis.flatten() - true_phis.flatten()) < eps
                        param_match = len(np.where(matches)[0]) / len(learned_phis.flatten())

                        param_rmse = np.sqrt(np.mean((learned_phis.flatten() - true_phis.flatten())**2))

                        df_params = df_params.append(
                            {
                                'Param Match': param_match,
                                'Param RMSE': param_rmse,
                                'Limit': limit,
                                'SSP Dim': ssp_dim,
                                'Coord Dim': coord_dim,
                            },
                            ignore_index=True
                        )
                except:
                    print("File not found: {}".format(fname_base.format(coord_dim, ssp_dim, limit)))
                    continue

    for limit in limits:
        try:
            df_tmp = df[df['Limit'] == limit]
            plt.figure()
            sns.barplot(data=df_tmp, x='SSP Dim', y='Test Loss', hue='Coord Dim')
            plt.title("Limit: {}".format(limit))

            df_tmp_params = df_params[df_params['Limit'] == limit]
            plt.figure()
            sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param Match', hue='Coord Dim')
            plt.title("Limit: {}".format(limit))

            plt.figure()
            sns.barplot(data=df_tmp_params, x='SSP Dim', y='Param RMSE', hue='Coord Dim')
            plt.title("Limit: {}".format(limit))
        except:
            print("no data for limit: {}".format(limit))
            continue

plt.show()
