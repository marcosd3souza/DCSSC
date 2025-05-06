import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

class NeighborhoodGraph:
    def __init__(self, X, D=None, verbose=False):
        if D is None:
            self.D = pairwise_distances(X)
            self.X = np.sort(X, axis=0)
        else:
            self.D = D
        self.verbose = verbose
        self.errors = []
        if verbose:
            print(f'initial dispersion in D: {np.std(self.D.flatten())}')


    def similarity_graph(self, X, n_components, k, sigma=0.01):
        return kneighbors_graph(X, n_neighbors=k, mode='connectivity').toarray()
