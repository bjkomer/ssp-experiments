import numpy as np
import nengo_spa as spa
import matplotlib.pyplot as plt
import argparse
from ssp_navigation.utils.encodings import add_encoding_params, get_encoding_function, get_encoding_heatmap_vectors
from spatial_semantic_pointers.utils import encode_point, get_heatmap_vectors, ssp_to_loc_v

parser = argparse.ArgumentParser()

# parser.add_argument('--neurons-per-dim', type=int, default=25)
# parser.add_argument('--n-samples', type=int, default=100)
# parser.add_argument('--n-items', type=int, default=7)
parser.add_argument('--res', type=int, default=256)  # 512
parser.add_argument('--encoder-type', type=str, default='mixed', choices=['mixed', 'grid', 'band', 'place', 'random'])
parser.add_argument('--new-dataset', action='store_true', help='generate a new random dataset to evaluate on')
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--n-items', type=int, default=7)


parser = add_encoding_params(parser)

args = parser.parse_args()

args.spatial_encoding = 'sub-toroid-ssp'
args.dim = 256
args.new_dataset = True

enc_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)

X = spa.SemanticPointer(data=enc_func(1, 0))
Y = spa.SemanticPointer(data=enc_func(0, 1))

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

# heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, repr_dim, enc_func, normalize=False)


# load pytorch data to compare to
pytorch_fname = 'pytorch_cleanup_results.npz'
pytorch_pred_vec = np.load(pytorch_fname)['pred_vec']


cache_fname = 'neural_cleanup_dataset_{}.npz'.format(args.dim)
if args.new_dataset:
    # output_fname = 'neural_cleanup_output_test_{}_{}.npz'.format(args.encoder_type, args.dim)
    output_fname = 'neural_cleanup_output_test_{}_{}_{}items_{}samples.npz'.format(
        args.encoder_type, args.dim, args.n_items, args.n_samples
    )
else:
    # output_fname = 'neural_cleanup_output_{}_{}.npz'.format(args.encoder_type, args.dim)
    output_fname = 'neural_cleanup_output_{}_{}_{}items_{}samples.npz'.format(
        args.encoder_type, args.dim, args.n_items, args.n_samples
    )

cache_data = np.load(cache_fname)
output_data = np.load(output_fname)
coords = cache_data['coords']
noisy = cache_data['noisy_vectors']
clean_true = cache_data['clean_vectors']
clean_pred_over_time = output_data['clean_output']

# slice out relevant part of clean output

dt = 0.001
time_steps = clean_pred_over_time.shape[0]
print('time_steps', time_steps)
steps_per_item = 100
n_samples = int(time_steps / steps_per_item)
print('n_samples', n_samples)
clean_pred = np.zeros((n_samples, args.dim))
# clean_pred[:, :] = clean_pred_over_time[50::steps_per_item, :]
# take the mean from the middle of the estimate
for i in range(n_samples):
    clean_pred[i, :] = clean_pred_over_time[i*steps_per_item+10:i*steps_per_item+100, :].mean(axis=0)

truth_coord_pred = np.zeros((n_samples, 2))

truth_coord_pred[:, :] = ssp_to_loc_v(
    clean_true[:n_samples, :],
    heatmap_vectors, xs, ys
)
error = np.zeros((n_samples,))
for i in range(n_samples):
    error[i] = np.linalg.norm(truth_coord_pred[i, :] - coords[i, :])
n_close = len(np.where(error < .5)[0])
print("truth fraction within 0.5 is {}".format(1.0*n_close / n_samples))




noisy_coord_pred = np.zeros((n_samples, 2))

noisy_coord_pred[:, :] = ssp_to_loc_v(
    noisy[:n_samples, :],
    heatmap_vectors, xs, ys
)
error = np.zeros((n_samples,))
for i in range(n_samples):
    error[i] = np.linalg.norm(noisy_coord_pred[i, :] - coords[i, :])
n_close = len(np.where(error < .5)[0])
print("noisy fraction within 0.5 is {}".format(1.0*n_close / n_samples))
noisy_ssp_rmse = np.sqrt((np.linalg.norm(noisy[:n_samples, :] - clean_true[:n_samples, :], axis=1) ** 2).mean())
print("noisy ssp rmse is {}".format(noisy_ssp_rmse))
noisy_coord_rmse = np.sqrt((np.linalg.norm(noisy_coord_pred - coords[:n_samples, :], axis=1) ** 2).mean())
print("noisy coord rmse", noisy_coord_rmse)






coord_pred = np.zeros((n_samples, 2))
norm_pred = np.zeros((n_samples, clean_pred.shape[1]))
for i in range(n_samples):
    norm_pred[i, :] = clean_pred[i, :] / np.linalg.norm(clean_pred[i, :])
coord_pred[:, :] = ssp_to_loc_v(
    norm_pred[:, :],
    heatmap_vectors, xs, ys
)

error = np.zeros((n_samples,))
for i in range(n_samples):
    error[i] = np.linalg.norm(coord_pred[i, :] - coords[i, :])
print(np.max(error))

n_close = len(np.where(error < .5)[0])
print("fraction within 0.5 is {}".format(1.0*n_close / n_samples))
ssp_rmse = np.sqrt((np.linalg.norm(clean_pred[:n_samples, :] - clean_true[:n_samples, :], axis=1) ** 2).mean())
print("spiking ssp rmse is {}".format(ssp_rmse))

# coord_rmse = np.sqrt((error**2).mean())
coord_rmse = np.sqrt((np.linalg.norm(coord_pred - coords[:n_samples, :], axis=1) ** 2).mean())
print("coord rmse", coord_rmse)

pytorch_coord_pred = np.zeros((n_samples, 2))
pytorch_norm_pred = np.zeros((n_samples, pytorch_pred_vec.shape[1]))
for i in range(n_samples):
    pytorch_norm_pred[i, :] = pytorch_pred_vec[i, :] / np.linalg.norm(pytorch_pred_vec[i, :])
pytorch_coord_pred[:, :] = ssp_to_loc_v(
    pytorch_norm_pred[:n_samples, :],
    heatmap_vectors, xs, ys
)

print(np.linalg.norm(pytorch_coord_pred - coords[:n_samples, :], axis=1).shape)

error = np.zeros((n_samples,))
for i in range(n_samples):
    error[i] = np.linalg.norm(pytorch_coord_pred[i, :] - coords[i, :])
n_close = len(np.where(error < .5)[0])
print("pytorch fraction within 0.5 is {}".format(1.0*n_close / n_samples))
pytorch_ssp_rmse = np.sqrt((np.linalg.norm(pytorch_pred_vec[:n_samples, :] - clean_true[:n_samples, :], axis=1) ** 2).mean())
print("pytorch ssp rmse is {}".format(pytorch_ssp_rmse))


pytorch_coord_rmse = np.sqrt((np.linalg.norm(pytorch_coord_pred - coords[:n_samples, :], axis=1) ** 2).mean())
print("pytorch coord rmse", pytorch_coord_rmse)

# show some images ground truth, noisy, cleaned
# sample_indices = [7, 13, 28]
# sample_indices = [7, 13, 27]
# sample_indices = [7, 13, 33]
sample_indices = [7, 13, 23]

# sample_indices = [56, 79, 102, 134, 198, 232]
# sample_indices = [395, 396, 397, 398, 399, 400]

fig, ax = plt.subplots(len(sample_indices), 4, figsize=(8, 6), tight_layout=True)

for i, si, in enumerate(sample_indices):
    truth_vec = clean_true[si, :]
    noisy_vec = noisy[si, :]
    pred_vec = clean_pred[si, :] / np.linalg.norm(clean_pred[si, :])

    truth_sim = np.tensordot(truth_vec, heatmap_vectors, axes=([0], [2]))
    noisy_sim = np.tensordot(noisy_vec, heatmap_vectors, axes=([0], [2]))
    pred_sim = np.tensordot(pred_vec, heatmap_vectors, axes=([0], [2]))
    pytorch_pred_sim = np.tensordot(pytorch_pred_vec[si, :]/np.linalg.norm(pytorch_pred_vec[si, :]), heatmap_vectors, axes=([0], [2]))

    ax[i, 0].imshow(truth_sim, vmin=0, vmax=1)
    ax[i, 1].imshow(noisy_sim, vmin=0, vmax=1)
    ax[i, 2].imshow(pred_sim, vmin=0, vmax=1)
    ax[i, 3].imshow(pytorch_pred_sim, vmin=0, vmax=1)

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()
    ax[i, 2].set_axis_off()
    ax[i, 3].set_axis_off()

    if i == 0:
        ax[i, 0].set_title("Ground Truth")
        ax[i, 1].set_title("Noisy Input")
        ax[i, 2].set_title("Spiking Cleanup")
        ax[i, 3].set_title("ANN Cleanup")
    #     pass
    # rmse_spiking = np.round(np.sqrt((np.linalg.norm(coord_pred[si, :] - coords[si, :]) ** 2).mean()), 2)
    # rmse_pytorch = np.round(np.sqrt((np.linalg.norm(pytorch_coord_pred[si, :] - coords[si, :]) ** 2).mean()), 2)
    # ax[i, 0].set_title("Ground Truth: {}".format(np.round(coords[si, :], 2)))
    # ax[i, 1].set_title("Noisy Input: {}, {}".format(rmse_spiking, rmse_pytorch))
    # ax[i, 2].set_title("Spiking Cleanup: {}".format(np.round(coord_pred[si, :],2)))
    # ax[i, 3].set_title("ANN Cleanup: {}".format(np.round(pytorch_coord_pred[si, :],2)))

plt.show()
