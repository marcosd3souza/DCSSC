# pip install oct2py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from sklearn.neighbors import kneighbors_graph
# from sklearn.utils import shuffle

# from keras.datasets import mnist
import numpy as np
import scipy.io as sio
from oct2py import octave
import pandas as pd
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from methods.Python.EDESC.EDESC import EDESC_exec
from methods.Python.ODSC.ODSC import ODSC_exec
from src.factorization import MatrixFactorization
from src.subspace import SubspaceRepresentation
from src.clustering import ClusteringBenchmark

from methods.Python.ENSC import SelfRepresentation
import methods.Python.LRR.LRR as lrr
import methods.Python.SSC.SSC as ssc
import methods.Python.DSC_Net.main as dsc_net
import methods.Python.DASC.main as dasc


def run_methods(method, X, name, X_img, X_input, X_last_layer):
    octave.restart()
    Z = None

    if method == 'LRR_L1':
        Z = lrr.lrr_exec_l1(X)
    elif method == 'LRR_L2':
        Z = lrr.lrr_exec_l2(X)
    elif method == 'SSC':
        Z = ssc.scc_exec(X)
    elif method == 'ENSC':
        Z = SelfRepresentation().fit_self_representation(X)
    elif method == 'DSC_Net':
        if name == 'MNIST':
            X_img = np.reshape(X, (-1, 1, 28, 28))
        else:
            X_img = np.reshape(X, (-1, 1, 32, 32))
        Z = dsc_net.dsc_net_exec(X_img)
    elif method == 'DASC':
        Z = dasc.dasc_exec(X_img, X_input, name)
    elif method == 'EDSC':
        octave.addpath('./methods/MATLAB/EDSC')
        Z = octave.edsc(X.T)

        # plt.imshow(Z)
        # plt.show()

    elif method == 'BDR':
        octave.addpath('./methods/MATLAB/BDR')
        Z = octave.bdr(X.T)
    elif method == 'PARTY':
        from methods.Python.PARTY.dsc import DeepSubspaceClustering
        model = DeepSubspaceClustering(inputX=X)
        model.train()
        Z = model.result
    elif method == 'ODSC':
        Z = ODSC_exec(X_img, X_input[0:2], X_last_layer)
    elif method == 'EDESC':
        Z = EDESC_exec(X).detach().numpy()
    # -------------------------- OURS
    elif method == 'M-DSC_Vae':
        # D = pairwise_distances(X)
        # G = kneighbors_graph(D, n_neighbors=5, mode='connectivity').toarray()
        #
        # plt.imshow(G)
        # plt.show()

        D_factored, _ = MatrixFactorization(X).NMF()
        # plt.imshow(D_factored)
        # plt.show()

        # Z = D_factored
        Z = SubspaceRepresentation(D_factored).vae_transform()
        # Z = Z.dot(Z.T)
        # plt.imshow(D_new)
        # plt.show()

        # Z, _ = MatrixFactorization(D_new).NMF()
        # Z = D * Z

        # Z = np.nan_to_num(Z)
        # Z = np.sort(Z, axis=0)
        # Z = kneighbors_graph(Z, n_neighbors=5).toarray()

        # plt.imshow(Z)
        # plt.show()
    elif method == 'T-DSC_Vae':
        D, _ = MatrixFactorization(X).NTD()

        # plt.imshow(D)
        # plt.show()

        Z = SubspaceRepresentation(D).vae_transform()

        # plt.imshow(Z)
        # plt.show()

    # plt.figure()
    # plt.imshow(Z)
    # plt.savefig(f'results/figs/Z_{name}_{std}.png')

    # if name == 'MNIST':
    #     plt.figure()
    #     plt.imshow(Z)
    #     plt.title(f'{method} (std={np.round(np.std(Z), 3)})')
    #     plt.savefig(f'results/figs/Z_{method}_{name}.png')
    # X_new = SubspaceRepresentation(Z).eig_transform(n_vals=100)
    # X_reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_new)
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true)
    # plt.title(f'10 clusters (std=50)')
    # plt.savefig(f'results/tsne_50_nmf_vae.png')
    # plt.show()

    # X_reduced = umap.UMAP(metric='precomputed').fit_transform(Z)
    # plt.figure()
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true)
    # plt.title(f'10 clusters (name={data_name})')
    # plt.savefig(f'results/figs/Z_{method}_{name}.png')

    Z = np.nan_to_num(Z)
    Z = np.sort(Z, axis=0)
    Z = pairwise_distances(Z)
    # #
    # plt.imshow(Z)
    # plt.show()

    return Z


# http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL100.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/YaleB/YaleB_32x32.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/ORL/ORL_32x32.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/MNIST/2k2k.mat
def _get_real_data():
    # MNIST
    data = sio.loadmat('./datasets/MNIST_2k2k.mat')
    mnist_df = data['fea']
    mnist_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    mnist_img = np.reshape(mnist_df, (-1, 28, 28, 1))
    mnist_input = [28, 28, 1]
    mnist_last_layer = 56

    # COIL20
    data = sio.loadmat('./datasets/COIL20.mat')
    coil20_df = data['fea']
    coil20_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    coil20_img = np.reshape(coil20_df, (-1, 32, 32, 1))
    coil20_input = [32, 32, 1]
    coil20_last_layer = 64

    # COIL100
    data = sio.loadmat('./datasets/COIL100.mat')
    coil100_df = data['fea']
    coil100_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    coil100_img = np.reshape(coil100_df, (-1, 32, 32, 1))
    coil100_input = [32, 32, 1]
    coil100_last_layer = 64

    # ORL
    data = sio.loadmat('./datasets/ORL_32x32.mat')
    orl_df = data['fea']
    orl_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    orl_img = np.reshape(orl_df, (-1, 32, 32, 1))
    orl_input = [32, 32, 1]
    orl_last_layer = 64

    # YaleB
    data = sio.loadmat('./datasets/YaleB_32x32.mat')
    YaleB_df = data['fea']
    YaleB_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    YaleB_img = np.reshape(YaleB_df, (-1, 32, 32, 1))
    YaleB_input = [32, 32, 1]
    YaleB_last_layer = 64

    return [
        (coil20_df, coil20_labels, coil20_img, coil20_input, coil20_last_layer, 'COIL20'),
        (mnist_df, mnist_labels, mnist_img, mnist_input, mnist_last_layer, 'MNIST'),
        (orl_df, orl_labels, orl_img, orl_input, orl_last_layer, 'ORL'),
        (YaleB_df, YaleB_labels, YaleB_img, YaleB_input, YaleB_last_layer, 'YaleB'),
        (coil100_df, coil100_labels, coil100_img, coil100_input, coil100_last_layer, 'COIL100')
    ]


def _get_simulated_data(k=10):
    n_samples = 500

    # std=5
    X_0, y_0_true = make_blobs(n_samples=n_samples, cluster_std=5.0, centers=k, n_features=64)

    X_reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_0)
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_0_true)
    plt.title(f'10 clusters (std=5.0)')
    plt.savefig(f'results/figs/baseline_5.png')

    _sorted = np.sort(np.concatenate([np.array(y_0_true)[:, None], X_0], axis=1), axis=0)
    X_0 = _sorted[:, 1:]
    y_0_true = _sorted[:, 0]

    # std=10
    X_1, y_1_true = make_blobs(n_samples=n_samples, cluster_std=10.0, centers=k, n_features=64)
    X_reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_1)
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_1_true)
    plt.title(f'10 clusters (std=10.0)')
    plt.savefig(f'results/figs/baseline_10.png')

    _sorted = np.sort(np.concatenate([np.array(y_1_true)[:, None], X_1], axis=1), axis=0)
    X_1 = _sorted[:, 1:]
    y_1_true = _sorted[:, 0]

    # std=15
    X_2, y_2_true = make_blobs(n_samples=n_samples, cluster_std=15.0, centers=k, n_features=64)
    X_reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_2)
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_2_true)
    plt.title(f'10 clusters (std=15.0)')
    plt.savefig(f'results/figs/baseline_15.png')

    _sorted = np.sort(np.concatenate([np.array(y_2_true)[:, None], X_2], axis=1), axis=0)
    X_2 = _sorted[:, 1:]
    y_2_true = _sorted[:, 0]

    return [(X_0, y_0_true, 5.0), (X_1, y_1_true, 10.0), (X_2, y_2_true, 15.0)]


if __name__ == "__main__":
    methods = [
        # 'LRR_L1',  # 1) 2012 (Python)
        # 'LRR_L2',  # 2) 2012 (Python)
        'SSC',  # 3) 2013 (Python)
        # 'EDSC',  # 4) 2014 (Matlab)
        # 'ENSC',  # 5) 2016 (Python)
        # 'DSC_Net',  # 7) 2017 (Python)
        # 'DASC', # 8) 2018 (Python)
        # 'BDR',  # 9) 2018 (Matlab)
        # 'T-DSC_Vae', # ours
        # 'M-DSC_Vae',  # ours
        # 'PARTY',  # 6) 2016 (Python) # this disable TF v2. Should be the last to call !!!
        # 'ODSC', # 10) 2021 (Python) # this disable TF v2. Should be the last to call !!!
        # 'EDESC' # 11) 2022 (Python)
    ]

    # stds = [5, 10, 15]
    result_df = {
        'name': [],
        'method_name': [],
        'acc': [],
        'nmi': [],
        'ari': []
        # 'y_true': [],
        # 'y_pred': []
    }

    n_repeat = 30
    # k = 10
    # datasets = _get_simulated_data(k)
    datasets = _get_real_data()

    for method in methods:
        print(f'------------------------------ {method} -------------------------------')
        for X, y_true, X_img, X_input, X_last_layer, name in datasets:
            print(f'------------------------------ data: {name} -------------------------------')

            # X = shuffle(X)
            k = len(np.unique(y_true))
            Z = run_methods(method, X, name, X_img, X_input, X_last_layer)
            affinity = 'precomputed_nearest_neighbors'

            # result_df['y_true'].append(y_true)
            # result_df['y_pred'].append(y_pred)

            for i in range(n_repeat):
                # print(f'************************************************* iter: {i}')
                model = ClusteringBenchmark(Z, k, affinity)
                y_pred = model.y_predict
                sil, acc, nmi, ari = model.evaluate(y_true)

                result_df['name'].append(name)
                result_df['method_name'].append(method)
                result_df['acc'].append(acc)
                result_df['nmi'].append(nmi)
                result_df['ari'].append(ari)

            acc_mean = np.mean(result_df['acc'])
            acc_std = np.std(result_df['acc'])
            nmi_mean = np.mean(result_df['nmi'])
            nmi_std = np.std(result_df['nmi'])
            ari_mean = np.mean(result_df['ari'])
            ari_std = np.std(result_df['ari'])
            print(f'acc: {acc_mean} (mean) - {acc_std} (std)')
            print(f'nmi: {nmi_mean} (mean) - {nmi_std} (std)')
            print(f'ari: {ari_mean} (mean) - {ari_std} (std)')

    # pd.DataFrame(result_df).to_csv('benchmark_ours.csv', sep=';')
