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
    temp = df[[f'{indice}_baseline', f'{indice}_nmf', f'{indice}_our_nmf', f'{indice}_vae']]
    temp.columns = ['Baseline', 'NMF', 'NMF (our)', 'VAE']
    temp.boxplot(rot=20)
    plt.savefig(f'results/figs/{std}/box_{indice}.png')


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
    'ari_our_nmf': [],

    'acc_vae': [],
    'nmi_vae': [],
    'ari_vae': []
}

std_candi = [0.5, 1.0, 5.0]#, 50.0, 100.0]
k_candi = [10]
n_repeat = 1

for k in k_candi:
    for _std in std_candi:
        X, y_true = make_blobs(n_samples=1000, cluster_std=_std, centers=k, n_features=100)

        # noise = np.random.noncentral_chisquare(2, 100, (X.shape[0], X.shape[0]))
        # x_noise = np.random.noncentral_chisquare(3, 20, X.shape)
        # noise = np.abs(np.random.normal(0, 1, (X.shape[0], X.shape[0])))
        # x_noise = np.random.pareto(1000, (X.shape[0], X.shape[1]))
        # x_noise = np.random.rand(X.shape[0], X.shape[1])

        # X, y_true = make_circles(n_samples=2000, noise=_std, random_state=0)
        # X, y_true = make_classification(
        #     n_samples=1000, n_features=10000, n_redundant=0, n_informative=10, n_clusters_per_class=1, n_classes=k
        # )
        # x_noise = np.random.normal(0, 1000, X.shape)
        # X = X + x_noise

        plt.figure()
        X_reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true)
        # plt.scatter(X[:, 0], X[:, 1], c=y_true)
        plt.savefig(f'results/figs/{_std}/data_{_std}.eps', format='eps')

        _sorted = np.sort(np.concatenate([np.array(y_true)[:, None], X], axis=1), axis=0)
        X = _sorted[:, 1:]
        y_true = _sorted[:, 0]

        for _ in range(n_repeat):
            # X = shuffle(X)

            # X_sort = np.sort(X, axis=0)

            print('-------------------------> BASELINE')
            # X = np.sort(X, axis=0)
            D = pairwise_distances(X)

            plt.figure()
            plt.axis('off')

            img = plt.imshow(D, cmap='hot')
            plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/D_baseline.eps', format='eps')

            # D = D_init + noise

            # plt.figure()
            # img = plt.imshow(D, cmap='hot')
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_noise_baseline.png')

            # S = kneighbors_graph(D, n_neighbors=5, mode='connectivity').toarray()
            # D_S = pairwise_distances(S)
            # plt.figure()
            # img = plt.imshow(D_S, cmap='hot')
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_S_baseline.png')

            # X = np.arange(0, X.shape[0])
            # Y = np.arange(0, X.shape[0])
            # X, Y = np.meshgrid(X, Y)
            # Z = D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # surf = ax.plot_surface(X, Y, Z, cmap='hot', linewidth=0, antialiased=False)
            # # ax.set_zlim(-1.01, 1.01)
            #
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.savefig(f'results/figs/{_std}/D_3d_baseline.png')

            # D = np.abs(D + noise)
            # plt.figure()
            # img = plt.imshow(D, cmap='hot')
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_noise_baseline.png')
            #
            bench = ClusteringBenchmark(D, k)
            # # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # # plt.savefig(f'results/figs/{_std}/baseline.png')
            #

            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_baseline'].append(acc)
            bench_result['nmi_baseline'].append(nmi)
            bench_result['ari_baseline'].append(ari)

            print('-------------------------> NMF')
            W_NMF, loss = MatrixFactorization(None, D).NMF()
            # D_NMF = np.sort(D_NMF, axis=0)
            D_NMF = pairwise_distances(W_NMF)
            # plt.figure()
            # img = plt.imshow(D_NMF, cmap='hot')
            # plt.title(f'loss={np.round(loss, 2)}')
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_NMF.eps', format='eps')
            #
            # S = kneighbors_graph(W_NMF, n_neighbors=5, mode='connectivity').toarray()
            # D_S = pairwise_distances(S)
            # plt.figure()
            # img = plt.imshow(D_S, cmap='hot')
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_S_nmf.png')

            #
            bench = ClusteringBenchmark(D_NMF, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf.png')
            #
            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_nmf'].append(acc)
            bench_result['nmi_nmf'].append(nmi)
            bench_result['ari_nmf'].append(ari)

            print('-------------------------> our NMF')
            model = MatrixFactorization(None, D)
            W_nNMF, loss = model.nNMF()
            # W_nNMF = np.sort(W_nNMF, axis=0)
            D_nNMF = pairwise_distances(W_nNMF)
            # Z_vae = SubspaceRepresentation(D_nNMF).vae_transform()

            plt.figure()
            plt.plot(model.errors[0:10])
            plt.savefig(f'results/figs/{_std}/D_nNMF_losses.eps', format='eps')

            plt.figure()
            plt.axis('off')
            img = plt.imshow(D_nNMF, cmap='hot')
            # plt.title(f'loss={np.round(loss, 2)}')
            plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/D_nNMF.eps', format='eps')

            # S = kneighbors_graph(W_nNMF, n_neighbors=5, mode='connectivity').toarray()
            # D_S = pairwise_distances(S)
            # plt.figure()
            # img = plt.imshow(D_S)
            # plt.colorbar(img, orientation='vertical')
            # plt.savefig(f'results/figs/{_std}/D_S_nNMF.png')

            bench = ClusteringBenchmark(D_nNMF, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf_vae.png')

            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_our_nmf'].append(acc)
            bench_result['nmi_our_nmf'].append(nmi)
            bench_result['ari_our_nmf'].append(ari)

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
            plt.figure()
            plt.axis('off')
            img = plt.imshow(D_Z_vae, cmap='hot')
            # plt.title(f'loss={np.round(minLoss,2)}')
            plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/D_Z_vae.eps', format='eps')

            plt.figure()
            plt.axis('off')
            img = plt.imshow(S_nNMF)
            plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/S_nNMF.eps', format='eps')

            S_D_Z_vae = kneighbors_graph(D_Z_vae, n_neighbors=5, mode='connectivity').toarray()
            plt.figure()
            plt.axis('off')
            img = plt.imshow(S_D_Z_vae)
            plt.colorbar(img, orientation='vertical')
            plt.savefig(f'results/figs/{_std}/S_D_Z_vae.eps', format='eps')

            bench = ClusteringBenchmark(D_Z_vae, k)
            # plt.scatter(X[:, 0], X[:, 1], label=bench.y_predict)
            # plt.savefig(f'results/figs/{_std}/nmf_vae.png')

            acc, nmi, ari = bench.evaluate(y_true)
            bench_result['acc_vae'].append(acc)
            bench_result['nmi_vae'].append(nmi)
            bench_result['ari_vae'].append(ari)

        bench_df = pd.DataFrame(bench_result)
        # print(bench_df.T)
        # bench_df.to_csv(f"results/benchs/benchmark_std-{_std}.csv", sep=';')

        # _save_boxplot_bench(bench_df, _std, 'sil')
        _save_boxplot_bench(bench_df, _std, 'acc')
        _save_boxplot_bench(bench_df, _std, 'nmi')
        _save_boxplot_bench(bench_df, _std, 'ari')
