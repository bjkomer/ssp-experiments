# compute PCA on the place cell firing from trajectories of an agent moving through an environment
import matplotlib.pyploy
from sklearn.decomposition import PCA, NMF
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=20)

args = parser.parse_args()


pca = PCA(n_components=args.n_components)

pca.fit(X)
