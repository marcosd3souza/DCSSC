import numpy as np
from sklearn.metrics import mean_squared_error


class RobustNMF:

    def initialize_WH(self, X_shape, k):

        W = np.random.random((X_shape[0], k))
        H = np.random.random((k, X_shape[1]))

        return W, H

    def __init__(self, X, k, lamda, maxiter):

        self.X = X
        self.X_shape = X.shape
        self.k = k
        self.lamda = lamda
        self.maxiter = maxiter

        self.W, self.H = self.initialize_WH(X.shape, k)
        # self.S = np.zeros(X.shape)

    def update_S(self, S, X_shape, lamda):

        for i in range(X_shape[0]):
            for j in range(X_shape[1]):

                if S[i, j] > (lamda / 2):
                    S[i, j] -= (lamda / 2)
                elif S[i, j] < -(lamda / 2):
                    S[i, j] += (lamda / 2)
                else:
                    S[i, j] = 0.0

        return S

    def update_W(self, X, W, H, S):

        numerator = np.abs(np.dot(S - X, H.T)) - np.dot(S - X, H.T)
        denominator = 2 * np.dot(np.dot(W, H), H.T)

        return (numerator / denominator) * W

    def update_H(self, X, W, H, S):

        numerator = np.abs(np.dot(W.T, S - X)) - np.dot(W.T, S - X)
        denominator = 2 * np.dot(np.dot(W.T, W), H)

        return (numerator / denominator) * H

    def normalize(self, W, H):

        W_square = np.power(W, 2)
        norm = np.sum(W_square, axis=0)
        W = W / norm
        H_T = H.T
        H_T = H_T * norm
        H = H_T.T
        return W, H

    def fit(self):

        X = self.X
        W = self.W
        H = self.H

        maxiter = self.maxiter
        lamda = self.lamda

        for iter in range(maxiter):
            S = X - np.dot(W, H)
            S = self.update_S(S, X.shape, lamda)

            W = self.update_W(X, W, H, S)
            H = self.update_H(X, W, H, S)

            W, H = self.normalize(W, H)

        self.W = W
        self.H = H
        self.S = S

    def rmse(self):

        X = self.X
        W = self.W
        H = self.H
        S = self.S

        rmse = np.sqrt(mean_squared_error(X, np.dot(W, H) + S))

        return rmse

