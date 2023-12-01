from sklearn.manifold import TSNE
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
    temp = df[[f'{indice}_baseline', f'{indice}_nmf', f'{indice}_our_nmf']]#, f'{indice}_vae']]
    temp.columns = ['Baseline', 'NMF', 'L2 NMF']#, 'VAE']
    temp.boxplot(rot=20)
    plt.savefig(f'results/figs/{std}/box_{indice}.eps', format='eps')


# path = 'https://raw.githubusercontent.com/marcosd3souza/FSMethodology/master/train_datasets/Carcinom_dataset.csv'
# data = pd.read_csv(path, sep=' ')
# X = data.drop(['Y'], axis=1)
# y_true = data['Y']

bench_result = {
    'acc_baseline': [],
    'nmi_baseline': [],
    'ari_baseline': [],

    'acc_nmf': [],
    'nmi_nmf': [],
    'ari_nmf': [],

    'acc_our_nmf': [],
    'nmi_our_nmf': [],
    'ari_our_nmf': []
}

std_candi = [5.0]#, 1.0, 5.0]#, 1.0, 5.0]#, 50.0, 100.0]
k_candi = [10]
n_repeat = 30

for k in k_candi:
    for _std in std_candi:
        X, y_true = make_blobs(n_samples=1000, cluster_std=_std, centers=k, n_features=100)

        _sorted = np.sort(np.concatenate([np.array(y_true)[:, None], X], axis=1), axis=0)
        X = _sorted[:, 1:]
        y_true = _sorted[:, 0]

        for _ in range(n_repeat):
            # X = shuffle(X)

            # X_sort = np.sort(X, axis=0)

            print('-------------------------> BASELINE')
            # X = np.sort(X, axis=0)
            D = pairwise_distances(X)
            S = kneighbors_graph(D, n_neighbors=5, mode='connectivity').toarray()

            model = SubspaceRepresentation(S)
            Z = model.vae_transform()

            Z = np.nan_to_num(Z)
            # Z = Z.dot(Z.T)
            Z = np.sort(Z, axis=0)
            # Z[Z > np.mean(Z)] = 0
            D_Z_vae_baseline = pairwise_distances(Z)

            plt.figure()
            plt.axis('off')

            plt.imshow(D, cmap='hot')
            # plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/D_baseline.eps', format='eps', bbox_inches='tight', pad_inches=0)

            bench = ClusteringBenchmark(D_Z_vae_baseline, k)

            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_baseline'].append(acc)
            bench_result['nmi_baseline'].append(nmi)
            bench_result['ari_baseline'].append(ari)

            print('-------------------------> NMF')
            D_NMF = MatrixFactorization(None, D).NMF()
            # D_NMF = np.sort(D_NMF, axis=0)
            # D_NMF = pairwise_distances(W_NMF)

            S = kneighbors_graph(D_NMF, n_neighbors=5, mode='connectivity').toarray()
            model = SubspaceRepresentation(S)
            Z = model.vae_transform()
            Z = np.nan_to_num(Z)
            Z = np.sort(Z, axis=0)
            D_Z_vae_NMF = pairwise_distances(Z)

            #
            bench = ClusteringBenchmark(D_Z_vae_NMF, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf.png')
            #
            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_nmf'].append(acc)
            bench_result['nmi_nmf'].append(nmi)
            bench_result['ari_nmf'].append(ari)

            # print('-------------------------> our NMF')
            # model = MatrixFactorization(None, D)
            # W_nNMF, loss = model.nNMF()
            # # W_nNMF = np.sort(W_nNMF, axis=0)
            # D_nNMF = pairwise_distances(W_nNMF)
            # # Z_vae = SubspaceRepresentation(D_nNMF).vae_transform()
            #
            # plt.figure()
            # plt.plot(model.errors[0:10])
            # plt.savefig(f'results/figs/{_std}/D_nNMF_losses.eps', format='eps')
            #
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(D_nNMF, cmap='hot')
            # # plt.title(f'loss={np.round(loss, 2)}')
            # # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_nNMF.eps', format='eps', bbox_inches='tight', pad_inches=0)
            #
            # # S = kneighbors_graph(W_nNMF, n_neighbors=5, mode='connectivity').toarray()
            # # D_S = pairwise_distances(S)
            # # plt.figure()
            # # img = plt.imshow(D_S)
            # # plt.colorbar(img, orientation='vertical')
            # # plt.savefig(f'results/figs/{_std}/D_S_nNMF.png')
            #
            # bench = ClusteringBenchmark(D_nNMF, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/nmf_vae.png')
            #
            # acc, nmi, ari = bench.evaluate(y_true)
            # bench_result['acc_our_nmf'].append(acc)
            # bench_result['nmi_our_nmf'].append(nmi)
            # bench_result['ari_our_nmf'].append(ari)

            print('-------------------------> DGSSC')
            S_nNMF = MatrixFactorization(None, D).similarity_graph()
            # D_S = pairwise_distances(S)
            model = SubspaceRepresentation(S_nNMF)
            Z = model.vae_transform()

            plt.figure()
            plt.plot(model.losses['loss'])
            plt.savefig(f'results/figs/{_std}/vae_losses.eps', format='eps')

            minLoss = min(model.losses['loss'])

            Z = np.nan_to_num(Z)
            # Z = Z.dot(Z.T)
            Z = np.sort(Z, axis=0)
            # Z[Z > np.mean(Z)] = 0
            D_Z_vae = pairwise_distances(Z)
            # D_Z[D_Z > 1.0] = D_Z[D_Z > 1.0] * 10
            # Z = S * Z
            # Z = Z.dot(Z.T)

            # temp = Z.copy()
            ##temp[Z < 1.0] = temp[Z < 1.0] * 10000
            ##temp[(Z < 100.0) & (Z > 1.0)] = temp[(Z < 100.0) & (Z > 1.0)] * 100
            # temp[(Z < 0.5) & (Z > 0.1)] = 1000
            # temp[(Z < 50.0) & (Z > 10.0)] = 10
            # temp[(Z < 5.0) & (Z > 1.0)] = 100
            # temp[(Z < 10.0) & (Z > 5.0)] = 10
            # temp[(Z < 50.0) & (Z > 10.0)] = 1
            ##temp[Z > 100.0] = temp[Z > 100.0] * 10
            # plt.figure()
            # plt.axis('off')
            # img = plt.imshow(D_Z_vae, cmap='hot')
            # # plt.title(f'loss={np.round(minLoss,2)}')
            # # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_Z_vae.eps', format='eps')

            plt.figure()
            plt.axis('off')
            img = plt.imshow(S_nNMF)
            # plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/S_nNMF.eps', format='eps', bbox_inches='tight', pad_inches=0)

            S_D_Z_vae = kneighbors_graph(D_Z_vae, n_neighbors=5, mode='connectivity').toarray()
            plt.figure()
            plt.axis('off')
            img = plt.imshow(S_D_Z_vae)
            # plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/S_D_Z_vae.eps', format='eps', bbox_inches='tight', pad_inches=0)

            bench = ClusteringBenchmark(D_Z_vae, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf_vae.png')

            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_our_nmf'].append(acc)
            bench_result['nmi_our_nmf'].append(nmi)
            bench_result['ari_our_nmf'].append(ari)

        bench_df = pd.DataFrame(bench_result)
        # print(bench_df.T)
        # bench_df.to_csv(f"results/benchs/benchmark_std-{_std}.csv", sep=';')

        # _save_boxplot_bench(bench_df, _std, 'sil')
        _save_boxplot_bench(bench_df, _std, 'acc')
        _save_boxplot_bench(bench_df, _std, 'nmi')
        _save_boxplot_bench(bench_df, _std, 'ari')
