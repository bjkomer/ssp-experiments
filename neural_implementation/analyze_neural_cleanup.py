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

parser = add_encoding_params(parser)

args = parser.parse_args([])

args.spatial_encoding = 'sub-toroid-ssp'
args.dim = 256

enc_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)

X = spa.SemanticPointer(data=enc_func(1, 0))
Y = spa.SemanticPointer(data=enc_func(0, 1))

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

# heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, repr_dim, enc_func, normalize=False)

cache_fname = 'neural_cleanup_dataset_{}.npz'.format(args.dim)
output_fname = 'neural_cleanup_output_{}.npz'.format(args.dim)

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
    clean_pred[i, :] = clean_pred_over_time[i*steps_per_item+5:i*steps_per_item+95, :].mean(axis=0)
coord_pred = np.zeros((n_samples, 2))

coord_pred[:, :] = ssp_to_loc_v(
    clean_pred[:, :],
    heatmap_vectors, xs, ys
)

coord_rmse = np.sqrt((np.linalg.norm(coord_pred - coords[:n_samples, :], axis=1) ** 2).mean())
print("coord rmse", coord_rmse)

# show some images ground truth, noisy, cleaned
sample_indices = [7, 13, 28]

fig, ax = plt.subplots(3, 3, figsize=(6, 6), tight_layout=True)

for i, si, in enumerate(sample_indices):
    truth_vec = clean_true[si, :]
    noisy_vec = noisy[si, :]
    pred_vec = clean_pred[si, :]

    truth_sim = np.tensordot(truth_vec, heatmap_vectors, axes=([0], [2]))
    noisy_sim = np.tensordot(noisy_vec, heatmap_vectors, axes=([0], [2]))
    pred_sim = np.tensordot(pred_vec, heatmap_vectors, axes=([0], [2]))

    ax[i, 0].imshow(truth_sim, vmin=0, vmax=1)
    ax[i, 1].imshow(noisy_sim, vmin=0, vmax=1)
    ax[i, 2].imshow(pred_sim, vmin=0, vmax=1)

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()
    ax[i, 2].set_axis_off()

plt.show()
