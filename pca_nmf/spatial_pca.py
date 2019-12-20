# compute PCA on the place cell firing from trajectories of an agent moving through an environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF
import numpy as np
import argparse
from ssp_navigation.utils.encodings import get_encoding_function
from utils import get_activations, spatial_heatmap

parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=20)
parser.add_argument('--dataset', type=str,
                    default='../path_integration/data/path_integration_raw_trajectories_100t_15s_seed13.npz')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding'
                    ])
parser.add_argument('--res', type=int, default=128, help='resolution of the spatial heatmap')

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

data = np.load(args.dataset)

encoding_func, dim = get_encoding_function(args, limit_low=0, limit_high=2.2)

activations, flat_pos = get_activations(data=data, encoding_func=encoding_func, encoding_dim=dim)

pca = PCA(n_components=args.n_components)
# pca = NMF(n_components=args.n_components)

print("Fitting PCA")
pca.fit(activations)

print("Getting covariance")
covariance = pca.get_covariance()

print("Plotting covariance")
plt.figure()
plt.imshow(covariance)

# sns.clustermap(covariance, metric='correlation', method='centroid')
# sns.clustermap(covariance, metric='euclidean', method='centroid')
# sns.clustermap(covariance, metric='euclidean')
# sns.clustermap(covariance, metric='correlation')
sns.clustermap(covariance, metric='correlation', method='weighted')
sns.clustermap(covariance, metric='correlation', method='complete')

plt.show()

# print(pca.singular_values_)

print("Transforming with PCA")
transformed_activations = pca.transform(activations)

print(activations.shape)
print(transformed_activations.shape)

print("computing heatmap")
xs = np.linspace(0, 2.2, args.res)
ys = np.linspace(0, 2.2, args.res)
heatmap = spatial_heatmap(transformed_activations, flat_pos, xs, ys)

for n in range(args.n_components):
    plt.figure()
    plt.imshow(heatmap[n, :, :])
plt.show()
