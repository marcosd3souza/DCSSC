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


class MatrixFactorization:
    def __init__(self, X, verbose=False):
        self.X = np.sort(X, axis=0)
        self.D = pairwise_distances(X)
        self.verbose = verbose
        if verbose:
            print(f'initial dispersion in D: {np.std(self.D.flatten())}')

    def NMF(self, n_components=10):
        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=300
        )

        # S = kneighbors_graph(self.D, n_neighbors=5).toarray()
        D_control = self.D #kneighbors_graph(self.D, n_neighbors=5).toarray() #

        # L = sparse.csgraph.laplacian(csgraph=S, normed=True)
        # vals, vecs = np.linalg.eig(L)
        # vecs = vecs[:, np.argsort(vals)[0:20]].real
        # D_control = pairwise_distances(S)

        # plt.imshow(D_control)
        # plt.show()
        W = None
        H = None
        # print(f'std before: {np.std(D_control)}')
        # errors = []
        last_loss = 99999999
        threshold = 5.0
        for it in range(10):
            W = nmf.fit_transform(D_control)
            H = nmf.components_

            error = nmf.reconstruction_err_
            if (error / last_loss) > threshold:
                break

            last_loss = error
            # W = gaussian_filter(W, sigma=5.0) # np.sort(W.dot(H), axis=0)
            # H = gaussian_filter(H, sigma=5.0)
            # D_control = S + W.dot(H)
            D_control = np.linalg.norm(W, axis=1).reshape(-1, 1).dot(np.linalg.norm(H, axis=0).reshape(1, -1))

            # S = kneighbors_graph(D_control, n_neighbors=20).toarray()
            # D_control = pairwise_distances(D_control)
            # nmf_std = np.std(D_recon_NMF.flatten())
            # if self.verbose:
            #     print(f'dispersion by NMF: {nmf_std}')


            print(f'it: {it} - recon error: {error}')
            # print(f'std: {np.std(D_control)}')

        # D_control = np.sort(D_control, axis=0)
        D_new = np.sort(D_control, axis=0)
        S = kneighbors_graph(D_new, n_neighbors=5, mode='connectivity').toarray()
        # plt.plot(errors)
        # plt.show()
        # D_control = W.dot(H)
        # D_control = pairwise_distances(S)
        # D_control = S * D_control
        # D_control[D_control > 0.0] = 1
        return S, 0

    def NTD(self, n_ranks=5):
        D_control = self.D # kneighbors_graph(self.D, n_neighbors=5, metric='precomputed').toarray()
        S = kneighbors_graph(D_control, n_neighbors=5).toarray()

        for it in range(10):
            tensor = tl.tensor(D_control, dtype='float')
            (core, factors), errors = non_negative_tucker(tensor, rank=[n_ranks, n_ranks, n_ranks, n_ranks], return_errors=True)

            # non_negative_tucker(tensor, rank=(n_ranks, n_ranks), n_iter_max=300, return_errors=False)
            # core, factors = tucker(tl.tensor(self.X, dtype='float'), rank=[n_ranks, n_ranks])

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
        return D_control, 0
