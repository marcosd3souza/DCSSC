import numpy as np
import time

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering

# import matplotlib.pyplot as plt

from .solve_lrr import solve_lrr


def lrr_exec_l2(data):
    A = data
    X = pairwise_distances(A)
    # A = np.random.randn(100, 200)
    # X = np.random.randn(100, 100)
    lamb = 0.01

    # L2 NORM
    # print("solve min |Z|_* + lambda |E|_21 s.t. X = AZ + E by exact ALM ...")
    #
    # tic = time.time()
    # Z1, E1 = solve_lrr(X, A, lamb, reg=0, alm_type=0)
    # obj1 = np.sum(np.linalg.svd(Z1)[1]) + lamb * np.sum(np.sqrt(np.sum(E1 ** 2, 1)))
    # print("Elapsed time:", time.time() - tic)
    #
    # print("objective value=", obj1)

    print("solve min |Z|_* + lambda |E|_21 s.t. X = AZ + E by inexact ALM ...")

    tic = time.time()
    Z2, E2 = solve_lrr(X, A, lamb, reg=0, alm_type=1)
    # obj2 = np.sum(np.linalg.svd(Z2)[1]) + lamb * np.sum(np.sqrt(np.sum(E2 ** 2, 1)))
    print("Elapsed time:", time.time() - tic)
    # print("objective value=", obj2)

    # diff = np.max(np.abs(Z1 - Z2))
    #
    # print("### Warning: difference of the solution found by those two \
    #       approaches: |Z1 - Z2|_inf=%f" % diff)

    return np.dot(Z2.T, Z2)


def lrr_exec_l1(data):
    A = data
    X = pairwise_distances(A)
    # A = np.random.randn(100, 200)
    # X = np.random.randn(100, 100)
    lamb = 0.01

    # L1 NORM
    # print("solve min |Z|_* + lambda |E|_1 s.t. X = AZ + E by exact ALM ...")
    # tic = time.time()
    # Z1, E1 = solve_lrr(X, A, lamb, reg=1, alm_type=0)
    # obj1 = np.sum(np.linalg.svd(Z1)[1]) + lamb * np.sum(np.sqrt(np.sum(E1 ** 2,
    #                                                                    1)))
    # print("Elapsed time:", time.time() - tic)
    #
    # print("objective value=", obj1)

    print("solve min |Z|_* + lambda |E|_1 s.t. X = AZ + E by inexact ALM ...")
    tic = time.time()
    Z2, E2 = solve_lrr(X, A, lamb, reg=1, alm_type=1)
    # obj2 = np.sum(np.linalg.svd(Z2)[1]) + lamb * np.sum(np.sqrt(np.sum(E2 ** 2, 1)))
    print("Elapsed time:", time.time() - tic)
    # print("objective value=", obj2)

    # diff = np.max(np.abs(Z1 - Z2))

    # print("### Warning: difference of the solution found by those two\
    #       approaches: |Z1 - Z2|_inf=", diff)

    return np.dot(Z2.T, Z2)


if __name__ == "__main__":
    k = 30
    data, y_true = make_blobs(n_samples=150, cluster_std=0.1, centers=k, shuffle=True, n_features=500, random_state=0)
    # X = pairwise_distances(A)

    Z = lrr_exec_l2(data)
    y_predict = SpectralClustering(
        n_clusters=k,
        affinity="precomputed_nearest_neighbors",
        n_jobs=-1,
        random_state=0
    ).fit_predict(Z)

    nmi = normalized_mutual_info_score(y_true, y_predict)
    ari = adjusted_rand_score(y_true, y_predict)

    print(f'nmi: {nmi}')
    print(f'ari: {ari}')

    plt.imshow(Z)
    plt.show()
