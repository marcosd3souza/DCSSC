# from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from methods.Python.PARTY.supporting_files.helpers import init_layer_weight, GenBatch, optimize
from methods.Python.PARTY.supporting_files.nncomponents import DenseLayer
from methods.Python.PARTY.sp import getSparcityPrior
# from sp_blitzl1 import sparseCoefRecovery
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds

import numpy as np
# from supporting_files.nncomponents import *
# from supporting_files.helpers import *


class DeepSubspaceClustering:

    def __init__(self, inputX, C=None, lambda1=0, lambda2=0, activation='tanh', \
                weight_init='uniform', noise=None, learning_rate=0.1, optimizer='Adam', ro=8, K=10, d=4):

        self.noise = noise
        self.result = None
        self.losses = []

        # post process C
        self.ro = ro
        self.K = K
        self.d = d

        n_sample, n_feat = inputX.shape

        hidden_dims = [(n_sample*2),  n_sample, (n_sample*2)]

        # M must be a even number
        assert len(hidden_dims) % 2 == 1

        # Add the end layer
        hidden_dims.append(n_feat)

        # self.depth = len(dims)

        # This is not the symbolic variable of tensorflow, this is real!
        self.inputX = inputX

        if C is None:
            # Transpose the matrix first, and get the whole matrix of C
            # self.inputC = sparseCoefRecovery(inputX.T)
            self.inputC = getSparcityPrior(inputX.T, lambda1=0, lambda2=0, epochs=100, print_step=1)
        else:
            self.inputC = C

        self.C = tf.placeholder(dtype=tf.float32, shape=[None, None], name='C')

        self.hidden_layers = []
        self.X = self._add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X'))

        input_hidden = self.X
        weights, biases = init_layer_weight(hidden_dims, inputX, weight_init)

        # J3 regularization term
        J3_list = []
        for init_w, init_b in zip(weights, biases):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].w)))
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].b)))

        J3 = lambda2 * tf.add_n(J3_list)

        self.H_M = self.hidden_layers[-1].output
        # H(M/2) the output of the mid layer
        self.H_M_2 = self.hidden_layers[int((len(hidden_dims)-1)/2)].output

        # calculate loss J1
        # J1 = tf.nn.l2_loss(tf.sub(self.X, self.H_M))

        J1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.X, self.H_M))))

        # calculate loss J2
        J2 = lambda1 * tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.transpose(self.H_M_2), \
                                     tf.matmul(tf.transpose(self.H_M_2), self.C)))))

        self.cost = J1 + J2 + J3

        self.optimizer = optimize(self.cost, learning_rate, optimizer)

    def train(self, batch_size=100, epochs=10, print_step=1):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        batch_generator = GenBatch(self.inputX, C=self.inputC, batch_size=batch_size)
        n_batch = batch_generator.n_batch

        x_batch, c_batch = (None, None)

        for i in range(epochs):
            # x_batch, y_batch = get_batch(self.X_train, self.y_train, batch_size)
            batch_generator.resetIndex()
            for j in range(n_batch+1):
                x_batch, c_batch = batch_generator.get_batch()
                sess.run(self.optimizer, feed_dict={self.X: x_batch, self.C: c_batch})

            self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))

            if i % print_step == 0:
                print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        # for i in xrange(1, epochs+1):
        #     x_batch, c_batch = get_batch_XC(self.inputX, self.inputC, batch_size)  
        #     self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))      
        #     if i % print_step == 0:
        #         print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        self.result = self._post_process(sess.run(self.H_M_2, feed_dict={self.X: x_batch, self.C: c_batch}))

    def _add_noise(self, x):
        if self.noise is None:
            return x
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if self.noise == 'mask':
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp

    def _post_process(self, C):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        n = C.shape[0]
        C = 0.5 * (C + C.T)
        # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
        r = self.d * self.K + 1
        U, S, _ = svds(C, r, v0=np.ones(n))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** self.ro)
        L = L / L.max()
        L = np.sort(L, axis=0)
        L = 0.5 * (L + L.T)

        return L


if __name__ == "__main__":
    k = 10
    X, y_true = make_blobs(n_samples=150, cluster_std=0.1, centers=k, n_features=500, random_state=0)
    # y_true = np.sort(y_true)
    _sorted = np.sort(np.concatenate([np.array(y_true)[:, None], X], axis=1), axis=0)
    X = _sorted[:, 1:]
    y_true = _sorted[:, 0]

    model = DeepSubspaceClustering(inputX=X)
    model.train()

    Z = model.result
    # plt.imshow(Z)
    # plt.show()