# pip install oct2py
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from sklearn.neighbors import kneighbors_graph
# from sklearn.utils import shuffle

# from keras.datasets import mnist
import numpy as np
import scipy.io as sio
from oct2py import octave
import pandas as pd
# from sklearn.datasets import make_blobs, load_digits
from keras.datasets import cifar10
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, binarize

from methods.Python.EDESC.EDESC import EDESC_exec
from methods.Python.ODSC.ODSC import ODSC_exec
from src.factorization import MatrixFactorization
from src.subspace import SubspaceRepresentation
from src.clustering import ClusteringBenchmark

# from scipy.linalg import block_diag

# from methods.Python.ENSC import SelfRepresentation
import methods.Python.LRR.LRR as lrr
import methods.Python.SSC.SSC as ssc
import methods.Python.DSC_Net.main as dsc_net
import methods.Python.DASC.main as dasc


def fuzioned_distance_matrix(X, k):
    # 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
    # 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    # 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    # metrics = ["braycurtis", "chebyshev", "cosine", "euclidean", "cityblock", "yule"]
    # gaussian = np.random.normal(0., 1., (X.shape[0], X.shape[0]))
    Ds = []
    # for m in metrics:
    h = int(X.shape[0] / k)
    for i in range(1, 11):
        dist = squareform(pdist(X, metric='minkowski', p=i)) ** 2
    #     # dist_bin = 1-binarize(dist, threshold=0.5)
    #     dist, _ = MatrixFactorization(None, D=dist).NMF(n_components=1)
        Ds.append(dist)
    # D = dist + block_diag(*([np.ones((h, h), dtype='int32') * 1] * k))
    return np.sum(Ds, axis=0)


def run_methods(method, X, name, X_img, X_input, X_last_layer, nmf_r, knn_k, y_true):
    # octave.restart()
    # octave.timeout = 5
    Z = None

    image_datasets = [
        'CIFAR10',
        'USPS',
        'Yale',
        'WarpAR10P',
        'WarpPIE10P',
        'Pixraw10P',
        'COIL20',
        'MNIST',
        'ORL',
        'YaleB',
        'COIL100'
    ]

    if method == 'LRR_L1':
        Z = lrr.lrr_exec_l1(X)
    elif method == 'LRR_L2':
        Z = lrr.lrr_exec_l2(X)
    elif method == 'SSC':
        Z = ssc.scc_exec(X)
    # elif method == 'ENSC':
    #     Z = SelfRepresentation().fit_self_representation(X)
    elif method == 'EDSC':
        octave.addpath('./methods/MATLAB/EDSC')
        Z = octave.edsc(X.T)
    elif method == 'BDR':
        octave.addpath('./methods/MATLAB/BDR')
        Z = octave.bdr(X.T)

    # -------------------------- OURS
    elif method == 'DGSSC_baseline':
        D = pairwise_distances(X)
        S = kneighbors_graph(D, n_neighbors=50, metric='precomputed').toarray()

        plt.imshow(S)
        plt.show()

        Z = SubspaceRepresentation(S).vae_transform()

    elif method == 'DGSSC':
        D = MinMaxScaler().fit_transform(pairwise_distances(X))
        # S = kneighbors_graph(D, n_neighbors=5).toarray()
        # D = fuzioned_distance_matrix(X, k)
        # Dn, error = MatrixFactorization(None, D).nNMF(n_components=5)
        # print(error)
        S = MatrixFactorization(None, D).similarity_graph(n_components=nmf_r, k=knn_k)

        # plt.gca().set_axis_off()
        # plt.imshow(Dn, cmap='hot')
        # plt.savefig(f'D_nNMF_rank_5.eps', format='eps', bbox_inches='tight')
        # plt.show()
        #
        # plt.gca().set_axis_off()
        # plt.imshow(S)
        # plt.savefig(f'S_D_nNMF_rank_5_k_150.eps', format='eps', bbox_inches='tight')
        # plt.show()

        Z = SubspaceRepresentation(S).cvae_transform(y_true)
    elif method == 'DGSSC_NMF':
        D = pairwise_distances(X)
        D_NMF = MatrixFactorization(None, D).NMF()
        S = kneighbors_graph(D_NMF, n_neighbors=5, mode='connectivity').toarray()

        Z = SubspaceRepresentation(S).vae_transform()

    elif name in image_datasets:
        if method == 'DSC_Net':
            if name == 'CIFAR10':
                X_img = np.reshape(X, (-1, 1, 32, 32))
            elif name == 'MNIST':
                X_img = np.reshape(X, (-1, 1, 28, 28))
            elif name == 'USPS':
                X_img = np.reshape(X, (-1, 1, 16, 16))
            elif name == 'WarpAR10P':
                X_img = np.reshape(X, (-1, 1, 60, 40))
            elif name == 'Pixraw10P':
                X_img = np.reshape(X, (-1, 1, 100, 100))
            else:
                X_img = np.reshape(X, (-1, 1, 32, 32))
            Z = dsc_net.dsc_net_exec(X_img)
        elif method == 'DASC':
            Z = dasc.dasc_exec(X_img, X_input, name)
        elif method == 'PARTY':
            from methods.Python.PARTY.dsc import DeepSubspaceClustering
            model = DeepSubspaceClustering(inputX=X)
            model.train()
            Z = model.result
        elif method == 'ODSC':
            Z = ODSC_exec(X_img, X_input[0:2], X_last_layer)
        elif method == 'EDESC':
            Z = EDESC_exec(X).detach().numpy()

    if Z is not None:
        Z = np.nan_to_num(Z)
        Z = np.sort(Z, axis=0)
        Z = pairwise_distances(Z)
    # #
    # plt.imshow(Z)
    # plt.show()

    # plt.gca().set_axis_off()
    # plt.imshow(Z)
    # plt.savefig(f'D_Z_{method}_rank_5.eps', format='eps', bbox_inches='tight')
    # plt.show()

    return Z


def _get_synthetic_data():
    data = pd.read_csv('./datasets/subspace_synthetic_data.csv', sep=';')
    y_subspace = np.array(data.loc[:, 'Y'], dtype=int)
    X_subspace = data.drop(['Y'], axis=1).values

    X_blobs, y_blobs = make_blobs(n_samples=1000, cluster_std=0.5, centers=5, n_features=10)
    _sorted = np.sort(np.concatenate([np.array(y_blobs)[:, None], X_blobs], axis=1), axis=0)
    X_blobs = _sorted[:, 1:]
    y_blobs = _sorted[:, 0]

    return [(X_subspace, y_subspace, None, None, None, 'synthetic_subspace'),
            (X_blobs, y_blobs, None, None, None, 'synthetic_blobs')]


# http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL100.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/YaleB/YaleB_32x32.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/ORL/ORL_32x32.mat
# http://www.cad.zju.edu.cn/home/dengcai/Data/MNIST/2k2k.mat
def _get_real_data():
    datasets = []
    [datasets.append(data_tuple) for data_tuple in _get_image_data()]
    # [datasets.append(data_tuple) for data_tuple in _get_bio_data()]
    # [datasets.append(data_tuple) for data_tuple in _get_text_data()]
    return datasets


def _get_text_data():
    # 20newsgroup
    data = sio.loadmat('./datasets/20Newsgroups.mat')
    labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    idx = np.array([
        np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
    ]).flatten()
    newsgroup20_df = data['fea'][idx].toarray()
    newsgroup20_labels = labels[idx]

    print('newsgroup20')
    print(newsgroup20_df.shape)
    print(max(newsgroup20_labels))

    # Reuters
    data = sio.loadmat('./datasets/Reuters21578.mat')
    labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    idx = np.array([
        np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
    ]).flatten()
    reuters_df = data['fea'][idx].toarray()
    reuters_labels = labels[idx]

    print('reuters')
    print(reuters_df.shape)
    print(max(reuters_labels))

    # TDT2
    data = sio.loadmat('./datasets/TDT2.mat')
    labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    idx = np.array([
        np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
    ]).flatten()
    tdt2_df = data['fea'][idx].toarray()
    tdt2_labels = labels[idx]

    print('TDT2')
    print(tdt2_df.shape)
    print(max(tdt2_labels))

    # RCV1
    data = sio.loadmat('./datasets/RCV1_4Class.mat')
    labels = np.squeeze(data['gnd'].toarray()).astype(int)
    idx = np.array([
        np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
    ]).flatten()
    rcv1_df = data['fea'][idx].toarray()
    rcv1_labels = labels[idx]

    print('RCV1')
    print(rcv1_df.shape)
    print(max(rcv1_labels))

    return [
        (newsgroup20_df, newsgroup20_labels, None, None, None, 'newsgroup20'),
        (reuters_df, reuters_labels, None, None, None, 'Reuters'),
        (tdt2_df, tdt2_labels, None, None, None, 'TDT2'),
        (rcv1_df, rcv1_labels, None, None, None, 'RCV1'),
    ]


def _get_bio_data():
    # Carcinom
    data = sio.loadmat('./datasets/Carcinom.mat')
    carcinom_df = data['X']
    carcinom_labels = np.sort(np.squeeze(data['Y'] - data['Y'].min() + 1))

    # Prostate-GE
    data = sio.loadmat('./datasets/Prostate_GE.mat')
    prostate_df = data['X']
    prostate_labels = np.sort(np.squeeze(data['Y'] - data['Y'].min() + 1))


    # TOX171
    data = sio.loadmat('./datasets/TOX_171.mat')
    tox171_df = data['X']
    tox171_labels = np.sort(np.squeeze(data['Y'] - data['Y'].min() + 1))

    # lung
    data = sio.loadmat('./datasets/lung.mat')
    lung_df = data['X']
    lung_labels = np.sort(np.squeeze(data['Y'] - data['Y'].min() + 1))

    # lymphoma
    data = sio.loadmat('./datasets/lymphoma.mat')
    lymphoma_df = data['X']
    lymphoma_labels = np.sort(np.squeeze(data['Y'] - data['Y'].min() + 1))

    return [
        (carcinom_df, carcinom_labels, None, None, None, 'Carcinom'),
        (prostate_df, prostate_labels, None, None, None, 'Prostate_GE'),
        (tox171_df, tox171_labels, None, None, None, 'TOX171'),
        (lung_df, lung_labels, None, None, None, 'lung'),
        (lymphoma_df, lymphoma_labels, None, None, None, 'lymphoma')
    ]


def _get_image_data():
    #CIFAR10
    (train_images, train_labels), _ = cifar10.load_data()
    labels = np.squeeze(train_labels)
    idx = np.array([np.random.choice(np.where(labels == l)[0], 100) for l in np.unique(labels)]).flatten()
    cifar10_df = train_images[idx]
    # grayscale
    cifar10_df = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifar10_df])
    cifar10_df = cifar10_df.reshape(1000, 1024)
    #
    cifar10_labels = labels[idx]
    # reshape
    cifar10_img = np.reshape(cifar10_df, (-1, 32, 32, 1))
    cifar10_input = [32, 32, 1]
    cifar10_last_layer = 64

    # MNIST
    data = sio.loadmat('./datasets/MNIST_2k2k.mat')
    mnist_df = data['fea']
    mnist_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    mnist_img = np.reshape(mnist_df, (-1, 28, 28, 1))
    mnist_input = [28, 28, 1]
    mnist_last_layer = 56

    # USPS
    data = sio.loadmat('./datasets/USPS.mat')
    labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    idx = np.array([
        np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
    ]).flatten()
    usps_df = data['fea'][idx]
    usps_labels = labels[idx]
    # reshape
    usps_img = np.reshape(usps_df, (-1, 16, 16, 1))
    usps_input = [16, 16, 1]
    usps_last_layer = 32

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

    # Yale
    data = sio.loadmat('./datasets/Yale_32x32.mat')
    Yale_df = data['fea']
    Yale_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
    # reshape
    Yale_img = np.reshape(Yale_df, (-1, 32, 32, 1))
    Yale_input = [32, 32, 1]
    Yale_last_layer = 64

    # warpAR10P
    data = sio.loadmat('./datasets/warpAR10P.mat')
    warpAR_df = data['X']
    warpAR_labels = np.squeeze(data['Y'] - data['Y'].min() + 1)
    # warpAR_labels = data['Y']
    # reshape
    warpAR_img = np.reshape(warpAR_df, (-1, 60, 40, 1))
    warpAR_input = [60, 40, 1]
    warpAR_last_layer = 80

    # pixraw10P
    data = sio.loadmat('./datasets/pixraw10P.mat')
    pixraw10P_df = data['X']
    pixraw10P_labels = np.squeeze(data['Y'] - data['Y'].min() + 1)
    # reshape
    pixraw10P_img = np.reshape(pixraw10P_df, (-1, 100, 100, 1))
    pixraw10P_input = [100, 100, 1]
    pixraw10P_last_layer = 200

    return [
        # (cifar10_df, cifar10_labels, cifar10_img, cifar10_input, cifar10_last_layer, 'CIFAR10'),
        # (usps_df, usps_labels, usps_img, usps_input, usps_last_layer, 'USPS'),
        # (Yale_df, Yale_labels, Yale_img, Yale_input, Yale_last_layer, 'Yale'),
        # (warpAR_df, warpAR_labels, warpAR_img, warpAR_input, warpAR_last_layer, 'WarpAR10P'),
        (pixraw10P_df, pixraw10P_labels, pixraw10P_img, pixraw10P_input, pixraw10P_last_layer, 'Pixraw10P'),
        # (coil20_df, coil20_labels, coil20_img, coil20_input, coil20_last_layer, 'COIL20'),
        # (mnist_df, mnist_labels, mnist_img, mnist_input, mnist_last_layer, 'MNIST'),
        # (orl_df, orl_labels, orl_img, orl_input, orl_last_layer, 'ORL'),
        # (YaleB_df, YaleB_labels, YaleB_img, YaleB_input, YaleB_last_layer, 'YaleB'),
        # (coil100_df, coil100_labels, coil100_img, coil100_input, coil100_last_layer, 'COIL100')
    ]


if __name__ == "__main__":
    methods = [
        # 'LRR_L1',  # 1) 2012 (Python)
        # 'LRR_L2',  # 2) 2012 (Python)
        # 'SSC',  # 3) 2013 (Python)
        # 'EDSC',  # 4) 2014 (Matlab)
        # 'ENSC',  # 5) 2016 (Python)
        # 'DSC_Net',  # 7) 2017 (Python)
        # 'DASC', # 8) 2018 (Python)
        # 'BDR',  # 9) 2018 (Matlab)
        # 'T-DSC_Vae', # ours
        # 'DGSSC_baseline',  # ours
        # 'DGSSC_NMF',  # ours
        'DGSSC',  # ours
        # 'PARTY',  # 6) 2016 (Python) # this disable TF v2. Should be the last to call !!!
        # 'ODSC', # 10) 2021 (Python) # this disable TF v2. Should be the last to call !!!
        # 'EDESC' # 11) 2022 (Python)
    ]

    # stds = [5, 10, 15]
    result_df = {
        'name': [],
        'method_name': [],
        'knn_k': [],
        'nmf_r': [],
        'acc': [],
        'nmi': [],
        'ari': []
        # 'y_true': [],
        # 'y_pred': []
    }

    knn_k_candi = [5] # range(5, 55, 5)
    nmf_r_candi = [5] # range(3, 50)
    n_repeat = 30
    datasets = _get_real_data()
    # datasets = _get_synthetic_data()

    for method in methods:
        print(f'------------------------------ {method} -------------------------------')
        for X, y_true, X_img, X_input, X_last_layer, name in datasets:
            print(f'------------------------------ data: {name} -------------------------------')

            # X = shuffle(X)
            k = len(np.unique(y_true))

            for knn_k in knn_k_candi:
                for nmf_r in nmf_r_candi:
                    for i in range(n_repeat):
                        Z = run_methods(method, X, name, X_img, X_input, X_last_layer, nmf_r, knn_k, y_true)

                        # if Z is not None:
                        #     affinity = 'precomputed_nearest_neighbors'

                        # print(f'************************************************* iter: {i}')
                        model = ClusteringBenchmark(Z, k)
                        y_pred = model.y_predict
                        acc, nmi, ari = model.evaluate(y_true)

                        result_df['name'].append(name)
                        result_df['method_name'].append(method)
                        result_df['knn_k'].append(knn_k)
                        result_df['nmf_r'].append(nmf_r)
                        result_df['acc'].append(acc)
                        result_df['nmi'].append(nmi)
                        result_df['ari'].append(ari)

                        # result_df['y_true'].append(y_true)
                        # result_df['y_pred'].append(y_pred)

            acc_mean = np.mean(result_df['acc'])
            acc_std = np.std(result_df['acc'])
            nmi_mean = np.mean(result_df['nmi'])
            nmi_std = np.std(result_df['nmi'])
            ari_mean = np.mean(result_df['ari'])
            ari_std = np.std(result_df['ari'])
            print(f'knn_k: {knn_k} - nmf_r: {nmf_r}')
            print(f'acc: {acc_mean} (mean) - {acc_std} (std)')
            print(f'nmi: {nmi_mean} (mean) - {nmi_std} (std)')
            print(f'ari: {ari_mean} (mean) - {ari_std} (std)')

    # pd.DataFrame(result_df).to_csv('params_sensibility.csv', sep=';')
    # pd.DataFrame(result_df).to_csv('benchmark_ours.csv', sep=';')
