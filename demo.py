from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs, make_circles, make_classification

from src.factorization import MatrixFactorization
from src.subspace import SubspaceRepresentation
from src.clustering import ClusteringBenchmark

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import pandas as pd
import numpy as np

from sklearn.neighbors import kneighbors_graph

def _save_boxplot_bench(df, std, indice):
    plt.figure()
    df.boxplot(column=[f'{indice}_baseline',
                       f'{indice}_nmf',     f'{indice}_ntd',
                       f'{indice}_nmf_eig', f'{indice}_ntd_eig',
                       f'{indice}_nmf_vae', f'{indice}_ntd_vae'], rot=20)
    plt.savefig(f'results/figs/{std}/box_{indice}.png')


# path = 'https://raw.githubusercontent.com/marcosd3souza/FSMethodology/master/train_datasets/Carcinom_dataset.csv'
# data = pd.read_csv(path, sep=' ')
# X = data.drop(['Y'], axis=1)
# y_true = data['Y']

bench_result = {
    'sil_baseline': [],
    'acc_baseline': [],
    'nmi_baseline': [],
    'ari_baseline': [],

    'sil_nmf': [],
    'acc_nmf': [],
    'nmi_nmf': [],
    'ari_nmf': [],

    'sil_ntd': [],
    'acc_ntd': [],
    'nmi_ntd': [],
    'ari_ntd': [],

    'sil_nmf_vae': [],
    'acc_nmf_vae': [],
    'nmi_nmf_vae': [],
    'ari_nmf_vae': [],

    'sil_nmf_eig': [],
    'acc_nmf_eig': [],
    'nmi_nmf_eig': [],
    'ari_nmf_eig': [],

    'sil_ntd_vae': [],
    'acc_ntd_vae': [],
    'nmi_ntd_vae': [],
    'ari_ntd_vae': [],

    'sil_ntd_eig': [],
    'acc_ntd_eig': [],
    'nmi_ntd_eig': [],
    'ari_ntd_eig': [],

}

std_candi = [50.0]  # 10.0, 5.0, 1.0]
k_candi = [10]
n_repeat = 1

for k in k_candi:
    for _std in std_candi:
        X, y_true = make_blobs(n_samples=500, cluster_std=_std, centers=k, n_features=64)
        # X, y_true = make_circles(n_samples=2000, noise=_std, random_state=0)
        # X, y_true = make_classification(
        #     n_samples=500, n_features=64, n_redundant=0, n_informative=10, n_clusters_per_class=1, n_classes=k
        # )
        _sorted = np.sort(np.concatenate([np.array(y_true)[:, None], X], axis=1), axis=0)
        X = _sorted[:, 1:]
        y_true = _sorted[:, 0]

        for _ in range(n_repeat):
            # X = shuffle(X)

            # X_sort = np.sort(X, axis=0)
            # D_baseline = pairwise_distances(X, metric='cosine')
            # _baseline_std = np.std(D_baseline)
            # plt.figure()
            # plt.imshow(D_baseline)
            # # plt.plot(X[:, 0], X[:, 1], label=)
            # plt.savefig(f'results/figs/{_std}/D_baseline.png')
            #
            # bench = ClusteringBenchmark(D_baseline, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/baseline.png')
            #
            # print('-------------------------> BASELINE')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_baseline'].append(sil)
            # bench_result['acc_baseline'].append(acc)
            # bench_result['nmi_baseline'].append(nmi)
            # bench_result['ari_baseline'].append(ari)
            #
            D_NTD, _ntd_std = MatrixFactorization(X).NTD()
            # plt.figure()
            # plt.imshow(D_NTD)
            # plt.savefig(f'results/figs/{_std}/D_ntd.png')
            # bench = ClusteringBenchmark(D_NTD, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/ntd.png')
            #
            # print('-------------------------> NTD')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_ntd'].append(sil)
            # bench_result['acc_ntd'].append(acc)
            # bench_result['nmi_ntd'].append(nmi)
            # bench_result['ari_ntd'].append(ari)
            #
            # D_NMF, _nmf_std = MatrixFactorization(X).NMF()
            # plt.figure()
            # plt.imshow(D_NMF)
            # plt.savefig(f'results/figs/{_std}/D_nmf.png')
            #
            # bench = ClusteringBenchmark(D_NMF, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/nmf.png')
            #
            # print('-------------------------> NMF')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_nmf'].append(sil)
            # bench_result['acc_nmf'].append(acc)
            # bench_result['nmi_nmf'].append(nmi)
            # bench_result['ari_nmf'].append(ari)

            # Z_nmf_vae = SubspaceRepresentation(D_NMF).vae_transform(y_true)
            # plt.figure()
            # plt.imshow(Z_nmf_vae)
            # plt.savefig(f'results/figs/{_std}/Z_nmf_vae.png')

            # bench = ClusteringBenchmark(Z_nmf_vae, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf_vae.png')

            # print('-------------------------> NMF VAE')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_nmf_vae'].append(sil)
            # bench_result['acc_nmf_vae'].append(acc)
            # bench_result['nmi_nmf_vae'].append(nmi)
            # bench_result['ari_nmf_vae'].append(ari)

            # Z_nmf_eig = SubspaceRepresentation(D_NMF).eig_transform()
            # plt.figure()
            # plt.imshow(Z_nmf_eig)
            # plt.savefig(f'results/figs/{_std}/Z_nmf_eig.png')
            #
            # bench = ClusteringBenchmark(Z_nmf_eig, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/nmf_eig.png')
            #
            # print('-------------------------> NMF EIG')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_nmf_eig'].append(sil)
            # bench_result['acc_nmf_eig'].append(acc)
            # bench_result['nmi_nmf_eig'].append(nmi)
            # bench_result['ari_nmf_eig'].append(ari)
            #
            Z_ntd_vae = SubspaceRepresentation(D_NTD).vae_transform(y_true)
            # plt.figure()
            # plt.imshow(Z_ntd_vae)
            # plt.savefig(f'results/figs/{_std}/Z_ntd_vae.png')
            # bench = ClusteringBenchmark(Z_ntd_vae, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/ntd_vae.png')
            #
            # print('-------------------------> NTD VAE')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_ntd_vae'].append(sil)
            # bench_result['acc_ntd_vae'].append(acc)
            # bench_result['nmi_ntd_vae'].append(nmi)
            # bench_result['ari_ntd_vae'].append(ari)
            #
            # Z_ntd_eig = SubspaceRepresentation(D_NTD).eig_transform()
            # plt.figure()
            # plt.imshow(Z_ntd_eig)
            # plt.savefig(f'results/figs/{_std}/Z_ntd_eig.png')
            #
            # bench = ClusteringBenchmark(Z_ntd_eig, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/ntd_eig.png')
            #
            # print('-------------------------> NTD EIG')
            # sil, acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['sil_ntd_eig'].append(sil)
            # bench_result['acc_ntd_eig'].append(acc)
            # bench_result['nmi_ntd_eig'].append(nmi)
            # bench_result['ari_ntd_eig'].append(ari)

        # bench_df = pd.DataFrame(bench_result)
        # print(bench_df.T)
        # bench_df.to_csv(f"results/benchs/benchmark_std-{_std}.csv", sep=';')

        # _save_boxplot_bench(bench_df, _std, 'sil')
        # _save_boxplot_bench(bench_df, _std, 'acc')
        # _save_boxplot_bench(bench_df, _std, 'nmi')
        # _save_boxplot_bench(bench_df, _std, 'ari')
