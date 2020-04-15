from ssp_navigation.utils.models import FeedForward
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import torch
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from ssp_navigation.utils.datasets import GenericDataset
from spatial_semantic_pointers.plots import plot_predictions_v

parser = argparse.ArgumentParser('View an untrained projection of an encoding')

parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--res', type=int, default=128)
parser = add_encoding_params(parser)

args = parser.parse_args()

encodings = [
    '2d',
    'ssp',
    'hex-ssp',
    'pc-gauss',
    'one-hot',
    'tile-coding',
    'legendre',
]

enc_names = {
    '2d': '2D',
    'ssp': 'SSP',
    'hex-ssp': 'Hex SSP',
    'pc-gauss': 'RBF',
    'one-hot': 'One-Hot',
    'tile-coding': 'Tile-Code',
    'legendre': 'Legendre',
    'random': 'Random',
}

torch.manual_seed(args.seed)
model = FeedForward(input_size=args.dim, hidden_size=args.hidden_size, output_size=2)

xs = np.linspace(-args.limit, args.limit, args.res)

outputs = np.zeros((args.res*args.res, 2))

for i, x in enumerate(xs):
    for j, y in enumerate(xs):
        outputs[i*args.res + j, :] = np.array([x, y])

for ei, enc in enumerate(encodings):
    print(enc)
    args.spatial_encoding = enc

    encoding_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)

    inputs = np.zeros((args.res * args.res, args.dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            # still use the full network for 2d, but just the first two elements are non-zero
            if enc == '2d':
                inputs[i * args.res + j, 0] = x
                inputs[i * args.res + j, 1] = y
            else:
                inputs[i * args.res + j, :] = encoding_func(x, y)

    dataset_test = GenericDataset(inputs=inputs, outputs=outputs)

    # For testing just do everything in one giant batch
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
    )

    with torch.no_grad():
        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):
            ssp, coord = data

            model_outputs = model(ssp)

        fig_pred, ax_pred = plt.subplots(tight_layout=True)
        plot_predictions_v(
            predictions=model_outputs, coords=coord,
            ax=ax_pred,
            min_val=-args.limit * 1.1,
            max_val=args.limit * 1.1,
            fixed_axes=False,
        )
        ax_pred.set_title(enc_names[enc])
        fig_pred.savefig("figures/untrained_{}_limit{}_dim{}.pdf".format(enc, int(args.limit), args.dim))
        # only record the ground truth once
        if ei == 0:
            fig_truth, ax_truth = plt.subplots(tight_layout=True)
            plot_predictions_v(
                predictions=coord, coords=coord,
                ax=ax_truth,
                min_val=-args.limit * 1.1,
                max_val=args.limit * 1.1,
                fixed_axes=False,
            )
            ax_truth.set_title("Coord Locations")
            fig_truth.savefig("figures/untrained_ground_truth_limit{}.pdf".format(args.limit))

# plt.show()
