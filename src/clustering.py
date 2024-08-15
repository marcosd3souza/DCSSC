import numpy as np

from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import SpectralClustering


class ClusteringBenchmark:
    def __init__(self, Z, k, affinity='precomputed_nearest_neighbors'):
        self.Z = Z # np.nan_to_num(Z)
        
        self.y_predict = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            random_state=np.random.randint(10000)
        ).fit_predict(self.Z)

    def evaluate(self, y_true):
        cm = metrics.confusion_matrix(y_true, self.y_predict)
        _make_cost_m = lambda x:-x + np.max(x)
        indexes = linear_assignment(_make_cost_m(cm))
        indexes = np.concatenate([indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
        js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
        cm2 = cm[:, js]
        acc = np.trace(cm2) / np.sum(cm2)

        # sil = metrics.silhouette_score(self.Z, self.y_predict, metric='euclidean')
        nmi = metrics.normalized_mutual_info_score(y_true, self.y_predict)
        ari = metrics.adjusted_rand_score(y_true, self.y_predict)

        print(f'acc: {acc}')
        print(f'nmi: {nmi}')
        print(f'ari: {ari}')

        return acc, nmi, ari
