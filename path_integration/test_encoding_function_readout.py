from path_integration_utils import encoding_func_from_model, pc_gauss_encoding_func
import numpy as np
from spatial_semantic_pointers.utils import ssp_to_loc_v, encode_point, make_good_unitary
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt

frozen_model_path = 'frozen_models/blocks_10_100_model.pt'

ssp_scaling = 5

dim = 512

limit_low = 0 * ssp_scaling
limit_high = 2.2 * ssp_scaling
res = 64 #128

# encoding = 'frozen-learned'
# encoding = 'ssp'
encoding = 'pc-gauss'

if encoding == 'frozen-learned':
    # Generate an encoding function from the model path
    encoding_func = encoding_func_from_model(frozen_model_path)
elif encoding == 'ssp':

    rng = np.random.RandomState(13)
    x_axis_sp = make_good_unitary(dim, rng=rng)
    y_axis_sp = make_good_unitary(dim, rng=rng)

    def encoding_func(coords):
        return encode_point(
            x=coords[0], y=coords[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
        ).v
elif encoding == 'pc-gauss':
    rng = np.random.RandomState(13)
    encoding_func = pc_gauss_encoding_func(limit_low=limit_low, limit_high=limit_high, dim=dim, rng=rng)


xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# encoding for every point in a 2D linspace, for approximating a readout

# FIXME: inefficient but will work for now
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


predictions = ssp_to_loc_v(
    flat_heatmap_vectors,
    heatmap_vectors, xs, ys
)

print(predictions)

coords = predictions.copy()

fig_pred, ax_pred = plt.subplots()


print("plotting predicted locations")
plot_predictions_v(predictions / ssp_scaling, coords / ssp_scaling, ax_pred, min_val=0, max_val=2.2)

plt.show()
