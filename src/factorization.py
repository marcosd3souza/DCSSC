import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
# from matplotlib import pyplot as plt
# from scipy import sparse
# from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances

# NTD
# import tensorly as tl
from sklearn.neighbors import kneighbors_graph
# from tensorly.decomposition import non_negative_tucker, tucker

#NMF
from sklearn.decomposition import NMF

from src import RNMF
from src.GAT import GAT


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

    def NMF(self, n_components=5, input=None):
        if input is None:
            input = self.D

        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=300
        )

        W = nmf.fit_transform(input)
        H = nmf.components_

        # error = nmf.reconstruction_err_

        return W.dot(H)

    def similarity_graph(self, X, n_components, k, sigma=0.01):

        # n_sample = self.D.shape[0]
        # s0 = kneighbors_graph(self.D, n_neighbors=k, mode='connectivity').toarray()
        # A = 1.0 / (self.D + np.identity(self.D.shape[0]))

        # sigma = np.median(self.D)
        # # Compute affinity matrix using Gaussian kernel (RBF)
        # A = np.exp(-self.D ** 2 / (2 * sigma ** 2))
        # S = A.dot(A.T)
        # plt.imshow(self.D)
        # plt.show()

        #
        # # # # Graph Laplacian.
        # Compute the normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
        # D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
        # L = np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
        # L = sparse.csgraph.laplacian(csgraph=A, normed=True)
        ## graph_laplacian = graph_laplacian_s.toarray()
        #
        # eigenvals, eigenvcts = np.linalg.eig(L)
        #
        # vals = eigenvals.real
        # vecs = eigenvcts.real
        #
        # positive = np.where(vals > 0)
        #
        # vals_index = np.argsort(vals)[0:10]
        # E = vecs[:, vals_index]
        #
        # E = np.sort(E, axis=0)

        # Generate a random matrix with Gaussian distribution
        # random_matrix = np.random.normal(loc=0., scale=1., size=(self.D.shape[0], self.D.shape[0]))
        # Make the matrix symmetric
        # E = (X_eig.dot(X_eig.T)) / 2

        # A = A + symmetric_matrix

        # Min-max normalization: (x - min) / (max - min)
        # matrix_min = np.min(E)
        # matrix_max = np.max(E)

        # S = (E - matrix_min) / (matrix_max - matrix_min)
        S = kneighbors_graph(X, n_neighbors=k, mode='connectivity').toarray()
        # S = self.NMF(n_components=n_components, input=S)
        # S = kneighbors_graph(S, n_neighbors=k, mode='connectivity').toarray()

        # best_D = pairwise_distances(best_W)

        # # Number of nodes and features
        # N = S.shape[0]  # Number of nodes
        # F_in = X.shape[1]  # Number of input features
        # F_out = N  # Number of output classes
        #
        # # Adjacency matrix (N x N)
        # adj = torch.tensor(S, dtype=torch.float32)
        #
        # X = torch.tensor(X, dtype=torch.float32)
        #
        # # Hyperparameters
        # n_hidden = 10  # Number of hidden units per head
        # n_heads = N  # Number of attention heads
        # dropout = 0.2  # Dropout rate
        # alpha = 0.2  # Negative slope for LeakyReLU
        #
        # # Initialize the GAT model
        # model = GAT(n_features=F_in, n_hidden=n_hidden, n_classes=F_out, dropout=dropout, alpha=alpha, n_heads=n_heads)
        #
        # # Forward pass to get the output and attention coefficients
        # output, attention_matrix = model(X, adj)
        #
        # # Threshold for creating new edges (e.g., 0.5)
        # threshold = 0.3
        # S = (attention_matrix > threshold).float().detach().numpy()
        #
        # plt.imshow(S)
        # plt.show()
        return S

    def nNMF(self, n_components=10):
        nmf = NMF(
            n_components=n_components,
            init='random',
            max_iter=500
        )

        D_control = self.D

        # best_D = None
        candidates = []
        for it in range(12):
            W = nmf.fit_transform(D_control)
            H = nmf.components_

            error = nmf.reconstruction_err_
            self.errors.append(error)
            candidates.append(W.dot(H))

            W_norm = np.linalg.norm(W, axis=1).reshape(-1, 1)
            H_norm = np.linalg.norm(H, axis=0).reshape(1, -1)

            # if error < best_loss:
            #     best_loss = error
            #     best_D = D_control
            # best_W = W

            D_control = W_norm.dot(H_norm) #W.dot(H)
            D_control[D_control < 0] = 0

            # print(f'it: {it} error: {error}')

            if error == np.inf:
                break

            print(f'NMF error: {error}')

            # print(f'it: {it} - recon error: {error}')
        # derivatives = [abs(self.errors[i] - self.errors[i-1]) - abs(self.errors[i+1] - self.errors[i]) for i in range(2,50)]
        # idx = np.where(np.array(derivatives) < 0)[0][1] + 2
        # print(f"best derivative: {idx}")
        idx = np.argmin(self.errors)
        best_D = candidates[idx]
        best_loss = self.errors[idx]

        # for i, v in enumerate(derivatives):
        #     if v < 0:
        #         best_D = candidates[i+2]
        #         break

        return best_D, best_loss
