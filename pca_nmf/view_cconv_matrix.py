# Viewing the circulant matrix for circular convolution of axis vectors
import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import power, encode_point, make_good_unitary
from ssp_navigation.utils.encodings import get_encoding_function
from utils import get_activations, spatial_heatmap
from sklearn.decomposition import PCA
from scipy.linalg import circulant
import nengo.spa as spa
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fname', type=str, default='output/rep_output.npz')
parser.add_argument('--n-components', type=int, default=20)
parser.add_argument('--dataset', type=str,
                    default='../path_integration/data/path_integration_raw_trajectories_100t_15s_seed13.npz')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding', 'pc-dog'
                    ])

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--pc-diff-sigma', type=float, default=1.0)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=512)

parser.add_argument('--res', type=int, default=256)
parser.add_argument('--limit-low', type=float, default=-5)
parser.add_argument('--limit-high', type=float, default=5)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

def circulant_matrix_to_vec(mat):

    dim = mat.shape[0]

    vec = np.zeros((dim,))

    for i in range(dim):
        for j in range(dim):
            vec[(i - j) % dim] += mat[i, j]

    vec /= dim

    return vec


rng = np.random.RandomState(seed=args.seed)

xs = np.linspace(args.limit_low, args.limit_high, args.res)

axis_vector_type = 'pca'

if axis_vector_type == 'standard':
    dim = 256
    X = make_good_unitary(dim, rng=rng)
    # X = spa.SemanticPointer(dim)
    # X.make_unitary()

    X_circ = circulant(X.v)
    X_vec = circulant_matrix_to_vec(X_circ)

    # assert (np.all(X_vec == X.v))
elif axis_vector_type == 'covariance':
    # generate SSP based on a given circulant matrix
    data = np.load(args.fname)
    X_circ = data['covariance']
    dim = X_circ.shape[0]
    X_vec = circulant_matrix_to_vec(X_circ)
    X = spa.SemanticPointer(data=X_vec)

    X.make_unitary()
    X_circ = circulant(X.v)

    print(X.v)
elif axis_vector_type == 'pca':
    data = np.load(args.dataset)

    # if the dataset already has activations, just load them
    if args.spatial_encoding in args.dataset:
        print("Loading activations directly")
        activations = data['activations']
        flat_pos = data['positions']
    else:
        print("Computing activations")
        encoding_func, dim = get_encoding_function(args, limit_low=0, limit_high=2.2)
        activations, flat_pos = get_activations(data=data, encoding_func=encoding_func, encoding_dim=dim)

    pca = PCA(n_components=args.n_components)
    # pca = NMF(n_components=args.n_components)

    print("Fitting PCA")
    pca.fit(activations)

    print("Getting covariance")
    covariance = pca.get_covariance()

    plt.figure()
    plt.imshow(covariance)

    X_vec = circulant_matrix_to_vec(covariance)
    X = spa.SemanticPointer(data=X_vec)

    # X.make_unitary()
    X_circ = circulant(X.v)
else:
    raise NotImplementedError



plt.figure()
plt.imshow(X_circ)

similarity = np.zeros((args.res,))
zero_vec = np.zeros((dim, ))
zero_vec[0] = 1

for i, x in enumerate(xs):
    p = power(X, x)
    similarity[i] = np.dot(p.v, zero_vec)

plt.figure()
plt.plot(similarity)

plt.show()

# def compute_circulant_matrix(vec):
#
#     mat = np.zeros((len(vec), len(vec)))
#
#     for i in range(len(vec)):
#         mat[i, :]



