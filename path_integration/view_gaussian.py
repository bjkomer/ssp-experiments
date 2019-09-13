import numpy as np
import matplotlib.pyplot as plt

seed = 13
ssp_scaling = 1
dim = 16#512
use_softmax = False
res = 64
limit_low = 0 * ssp_scaling
limit_high = 2.2 * ssp_scaling
pc_gauss_sigma = .1#.25

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def pc_gauss_encoding_func(limit_low=0, limit_high=1, dim=512, sigma=0.01, use_softmax=False, rng=np.random):
    # generate PC centers
    pc_centers = rng.uniform(low=limit_low, high=limit_high, size=(dim, 2))

    # TODO: make this more efficient
    def encoding_func(positions):
        activations = np.zeros((dim,))
        for i in range(dim):
            activations[i] = np.exp(-((pc_centers[i, 0] - positions[0]) ** 2 + (pc_centers[i, 1] - positions[1]) ** 2) / sigma / sigma)
        if use_softmax:
            return softmax(activations)
        else:
            return activations

    return encoding_func

rng = np.random.RandomState(seed)
encoding_func = pc_gauss_encoding_func(
    limit_low=limit_low, limit_high=limit_high, dim=dim, sigma=pc_gauss_sigma, rng=rng, use_softmax=use_softmax
)

heatmap_vectors = np.zeros((len(xs), len(ys), dim))

flat_heatmap_vectors = np.zeros((len(xs) * len(ys), dim))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        heatmap_vectors[i, j, :] = encoding_func(
            # batch dim
            # np.array(
            #     [[x, y]]
            # )
            # no batch dim
            np.array(
                [x, y]
            )
        )

        flat_heatmap_vectors[i*len(ys)+j, :] = heatmap_vectors[i, j, :].copy()

        # Normalize. This is required for frozen-learned to work
        heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

summation = heatmap_vectors.sum(axis=2)

plt.figure()
plt.imshow(summation)
plt.figure()
plt.imshow(heatmap_vectors[:, :, 0])
plt.figure()
plt.imshow(heatmap_vectors[:, :, 1])
plt.figure()
plt.imshow(heatmap_vectors[:, :, 2])
plt.show()

