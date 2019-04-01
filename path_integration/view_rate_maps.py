import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.interpolation import rotate
import os

# https://github.com/lsolanka/gridcells/blob/c18423426787edcdd45a4a5d9058ae0285c57eca/tests/unit/fields_ref_impl.py#L70
def SNAutoCorr(rate_map, arena_diameter, res):
    xedges = np.linspace(-arena_diameter, arena_diameter, res*2 + 1)
    yedges = np.linspace(-arena_diameter, arena_diameter, res*2 + 1)
    X, Y = np.meshgrid(xedges, yedges)

    corr = ma.masked_array(
        signal.correlate2d(rate_map, rate_map),
        mask=np.sqrt(X**2 + Y**2) > arena_diameter
    )

    return corr, xedges, yedges


def gridness(rate_maps, center_radius=2, xs, ys):
    # based on: https://github.com/lsolanka/gridcells/blob/master/gridcells/analysis/fields.py#L139
    angles = [0, 30, 60, 90, 120, 150]

    n_neurons = rate_maps.shape[0]
    n_x = rate_maps.shape[1]
    n_y = rate_maps.shape[2]
    size = (n_x*2-1) * (n_y*2-1)
    n_angles = len(angles)

    arena_diameter = xs[-1] - xs[0]
    res = len(xs)


    # Contains rotations of 0, 30, 60, 90, and 150 degrees
    # rot_map = np.zeros((n_angles, n_neurons, n_x, n_y))

    # Contains rotations of 0, 30, 60, 90, and 150 degrees
    corr_maps = np.zeros((n_angles, n_neurons, n_x*2 - 1, n_y*2 - 1))

    X, Y = np.meshgrid(xs, ys)

    for ni in range(n_neurons):
        print("Computing correlation of neuron {} of {}".format(ni+1, n_neurons))

        rate_map_mean = rate_maps[ni, :, :] - np.mean(rate_maps[ni, :, :])

        auto_corr, autoC_xedges, autoC_yedges = SNAutoCorr(rate_map_mean, arena_diameter, res)

        auto_corr[np.sqrt(X**2 + Y**2) < center_radius] = 0

        # corr = signal.correlate2d(
        #     rate_maps[ni, :, :],
        #     rate_maps[ni, :, :],
        #     mode='full',
        #     boundary='fill',
        #     fillvalue=0,
        # )
        #
        # # zero out the centers
        # corr[np.sqrt(X**2 + Y**2) < center_radius] = 0
        #
        # corr_maps[0, ni, :, :] = corr



    ccs = np.zeros((n_angles, n_neurons))

    rot_map[0, :, :, :] = rate_maps
    for i, ang in enumerate(angles):
        if i != 0:
            # rot_map[i, :, :, :] = rotate(rate_maps, ang, axes=(1, 2))
            corr_maps[i, :, :, :] = rotate(corr_maps[0, :, :, :], ang, axes=(1, 2), reshape=False)

        for ni in range(n_neurons):

            # cc = np.corrcoef(
            #     np.reshape(rot_map[0, ni, :, :], (1, size)),
            #     np.reshape(rot_map[i, ni, :, :], (1, size))
            # )
            cc = np.corrcoef(
                np.reshape(corr_maps[0, ni, :, :], (1, size)),
                np.reshape(corr_maps[i, ni, :, :], (1, size))
            )
            ccs[i, ni] = cc[0, 1]

    max_angs_i = [1, 3, 5]
    min_angs_i = [2, 4]

    print("Computing gridness scores")

    gridness_scores = np.zeros((n_neurons))

    for ni in range(n_neurons):
        maxima = np.max(ccs[max_angs_i, ni])
        minima = np.min(ccs[min_angs_i, ni])
        gridness_scores[ni] = minima - maxima

    return gridness_scores, ccs


data = np.load('output/rate_maps.npz')

rate_maps_pred=data['rate_maps_pred']
rate_maps_truth=data['rate_maps_truth']

n_neurons = rate_maps_pred.shape[0]
n_x = rate_maps_pred.shape[1]
n_y = rate_maps_pred.shape[2]

pred_corr_filename = 'output/cross_corr_pred.npz'
if os.path.isfile(pred_corr_filename):
    pred_corr_data = np.load(pred_corr_filename)
    gridness_scores_pred = pred_corr_data['gridness_scores']
    css_pred = pred_corr_data['cross_correlations']
else:

    print("Computing gridness scores for predicted location ratemaps")
    gridness_scores_pred, ccs_pred = gridness(rate_maps_pred, 2)

    np.savez(
        'output/cross_corr_pred.npz',
        gridness_scores=gridness_scores_pred,
        cross_correlations=ccs_pred,
    )

    print(gridness_scores_pred)

truth_corr_filename = 'output/cross_corr_truth.npz'
if os.path.isfile(truth_corr_filename):
    truth_corr_data = np.load(truth_corr_filename)
    gridness_scores_truth = truth_corr_data['gridness_scores']
    css_truth = truth_corr_data['cross_correlations']
else:

    print("Computing gridness scores for predicted location ratemaps")
    gridness_scores_truth, ccs_truth = gridness(rate_maps_truth, 2)

    np.savez(
        'output/cross_corr_pred.npz',
        gridness_scores=gridness_scores_truth,
        cross_correlations=ccs_truth,
    )

    print(gridness_scores_truth)



if False:

    vmin=None
    vmax=None

    # vmin=0
    # vmax=1


    n_rows = int(np.ceil(np.sqrt(rate_maps_pred.shape[0])))
    n_cols = n_rows

    fig_pred, ax_pred = plt.subplots(n_rows, n_cols)

    fig_corr, ax_corr = plt.subplots(n_rows, n_cols)

    ni = 0
    for i in range(n_rows):
        for j in range(n_cols):
            corr = signal.correlate2d(
                rate_maps_pred[ni, :, :],
                rate_maps_pred[ni, :, :],
                mode='full',
                boundary='fill',
                fillvalue=0,
            )
            ax_corr[i, j].imshow(corr)

            ax_pred[i, j].imshow(rate_maps_pred[ni, :, :], vmin=vmin, vmax=vmax)

            ni += 1
            if ni == rate_maps_pred.shape[0]:
                break
        if ni == rate_maps_pred.shape[0]:
            break

    plt.show()

    # for ni in range(rate_maps_pred.shape[0]):
    #     print("Neuron {} of {}".format(ni + 1, rate_maps_pred.shape[0]))
    #     corr = signal.correlate2d(
    #         rate_maps_pred[ni, :, :],
    #         rate_maps_pred[ni, :, :],
    #         mode='full',
    #         boundary='fill',
    #         fillvalue=0,
    #     )
    #     plt.imshow(corr)
    #     # plt.imshow(rate_maps_pred[ni, :, :], vmin=vmin, vmax=vmax)
    #     # plt.imshow(rate_maps_truth[ni, :, :], vmin=vmin, vmax=vmax)
    #     plt.show()
