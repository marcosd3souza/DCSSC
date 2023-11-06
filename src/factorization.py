import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances

# NTD
import tensorly as tl
from sklearn.neighbors import kneighbors_graph
from tensorly.decomposition import non_negative_tucker, tucker

#NMF
from sklearn.decomposition import NMF

from src import RNMF


class MatrixFactorization:
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

    def rNMF(self, n_components=10):
        rnmf = RNMF.RobustNMF(self.D, n_components, 2, 30)
        rnmf.fit()

        return rnmf.W.dot(rnmf.H) + rnmf.S

    def NMF(self, n_components=5):
        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=300
        )

        W = nmf.fit_transform(self.D)
        # H = nmf.components_

        error = nmf.reconstruction_err_

        return W, error#.dot(H)

    def similarity_graph(self):

        best_D, _ = self.nNMF()

        S = kneighbors_graph(best_D, n_neighbors=5, mode='connectivity').toarray()

        # best_D = pairwise_distances(best_W)
        return S

    def nNMF(self, n_components=10):
        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=300
        )

        # S = kneighbors_graph(self.D, n_neighbors=5).toarray()
        # noise = np.abs(np.random.normal(0, 100, (self.D.shape[0], self.D.shape[0])))
        D_control = self.D #kneighbors_graph(self.D, n_neighbors=5).toarray() #
        # best_D = D_control
        # L = sparse.csgraph.laplacian(csgraph=S, normed=True)
        # vals, vecs = np.linalg.eig(L)
        # vecs = vecs[:, np.argsort(vals)[0:20]].real
        # D_control = pairwise_distances(S)

        # S = kneighbors_graph(best_D, n_neighbors=5, mode='connectivity').toarray()

        # plt.imshow(D_control)
        # plt.show()

        best_D = None
        best_loss = np.inf
        for it in range(30):
            W = nmf.fit_transform(D_control)
            H = nmf.components_

            error = nmf.reconstruction_err_
            self.errors.append(error)

            W_norm = np.linalg.norm(W, axis=1).reshape(-1, 1)
            H_norm = np.linalg.norm(H, axis=0).reshape(1, -1)

            if error < best_loss:
                best_loss = error
                best_D = D_control
                # best_W = W

            D_control = W_norm.dot(H_norm) #+ (best_D - W_norm.dot(H_norm))

            if error == np.inf:
                break

            print(f'it: {it} - recon error: {error}')

        return best_D, best_loss

    def NTD(self, n_ranks=5):
        D_control = self.D # kneighbors_graph(self.D, n_neighbors=5, metric='precomputed').toarray()
        S = kneighbors_graph(D_control, n_neighbors=5).toarray()

        for it in range(10):
            tensor = tl.tensor(D_control, dtype='float')
            (core, factors), errors = non_negative_tucker(tensor, rank=[n_ranks, n_ranks, n_ranks, n_ranks], return_errors=True)

            # non_negative_tucker(tensor, rank=(n_ranks, n_ranks), n_iter_max=300, return_errors=False)
            # core, factors = tucker(tl.tensor(self.X, dtype='float'), rank=[n_ranks, n_ranks])
            self.errors.append(errors[0])

            D_control = tl.tucker_to_tensor((core, factors))
            #D_control = D_control ** (S + 10)
            # ntd_std = np.std(D_recon_NTD.flatten())
            # if self.verbose:
            #     print(f'dispersion by NTD: {ntd_std}')

            print(f'it {it} error {errors}')
        # self.D = D_control
        # D_control = pairwise_distances(D_control)
        D_control = S * D_control
        # D_control = kneighbors_graph(pairwise_distances(D_control), n_neighbors=5, metric='precomputed').toarray()
        return D_control
