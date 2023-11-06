import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from sklearn.manifold import TSNE
import umap

from sklearn.metrics import pairwise_distances

import keras
from keras import backend as K
from keras import losses
from keras.models import Model
from keras.layers import Lambda, Dense, Input, LeakyReLU, BatchNormalization, Softmax
from sklearn.neighbors import kneighbors_graph

import tensorflow as tf

class SubspaceRepresentation:
    def __init__(self, D_factorized):
        self.D_factorized = D_factorized
        self.losses = []
        # self.D_original = D_original
    
    def eig_transform(self, n_vals=10):
        # # # Graph Laplacian.
        L = sparse.csgraph.laplacian(csgraph=self.D_factorized, normed=True)
        # graph_laplacian = graph_laplacian_s.toarray()

        eigenvals, eigenvcts = np.linalg.eig(L)

        vals = eigenvals.real
        vecs = eigenvcts.real

        positive = np.where(vals>0)

        vals_index = np.argsort(vals[positive])[0:n_vals]
        temp = vecs[:, vals_index]

        X_eig = np.sort(temp, axis=0)

        return pairwise_distances(X_eig)
    
    def vae_transform(self, y=None):
        dims = [self.D_factorized.shape[0], 300, 100, 10]

        pretrain_epochs = 100
        batch_size = 256

        autoencoder, encoder, decoder = self._vae_model(dims)
        # D = pairwise_distances(X_new)

        autoencoder.compile(optimizer='adam')
        history = autoencoder.fit(
            self.D_factorized,
            self.D_factorized,
            batch_size=batch_size,
            epochs=pretrain_epochs
        )

        self.losses = history.history

        # Z = autoencoder.predict(self.D_factorized)
        # Z = np.nan_to_num(Z)
        # Z = np.sort(Z, axis=0)
        D = self.D_factorized
        # Z = encoder.predict(D)[2]

        # Z = D
        # #
        # for it in range(10):
        #     # D = autoencoder.predict(D)
        #     Z = encoder.predict(D)[2]
        #     Z = Z.dot(Z.T)
        #     D = D * Z

            # D = D * Z
        Z = encoder.predict(D)[2]
        # Z = Z.dot(Z.T) * pairwise_distances(Z)
        # Z = Z.dot(Z.T)
        # Z = kneighbors_graph(Z, n_neighbors=5, mode='connectivity').toarray()

        # # # Graph Laplacian.
        # L = sparse.csgraph.laplacian(csgraph=S, normed=True)
        # vals, vecs = np.linalg.eig(L)
        # Z = vecs.real
        # Z = vecs[:, np.argsort(vals)[1:10]].real
        # Z = np.sort(Z, axis=0)
        # Z = np.linalg.norm(Z, axis=1).reshape(-1, 1)
        # Z = Z.dot(Z.T)
        # Z = np.nan_to_num(Z)
        # Z = np.linalg.norm(Z, axis=1).reshape(-1, 1).dot(np.linalg.norm(Z, axis=1).reshape(1, -1))#Z.dot(Z.T)
        # Z = np.abs((D * Z))

        # Z = (Z + Z.T) / 2
        # Z = ((D + D.T) / 2) + (Z.dot(Z.T))#Z.dot(Z.T)
        # Z = pairwise_distances(Z)
        # Z = np.sort(Z, axis=0)
        # Z = kneighbors_graph(Z, n_neighbors=5, mode='connectivity').toarray()
        # Z[Z < 1.0] = 0
        # Z[Z > 1.0] = 1
        # Z = Z[Z > 0].reshape(Z.shape[0], 5)

        # X_reduced = TSNE(n_components=2, learning_rate='auto', init='random', metric='precomputed').fit_transform(Z)
        # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
        # plt.title(f'10 clusters (std=50)')
        # plt.savefig(f'results/tsne_50_nmf_vae.png')
        # plt.show()

        # X_reduced = umap.UMAP(metric='precomputed').fit_transform(Z)
        # plt.imshow(Z)
        # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
        # plt.title(f'10 clusters (std=50)')
        # plt.savefig(f'results/Z_nmf_vae.png')

        # return pd.DataFrame(Z).T.corr('pearson').values
        return Z

    def _sampling(self, args):
        z_mean, z_log_sigma, latent_dim = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0., stddev=0.01)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_model(self, dims):

        # STEP 1: CREATE ENCODER
        n_stacks = len(dims) - 1
        # input
        input_visible = Input(shape=(dims[0],), name='data_input')
        x = input_visible

        # knn graph
        # G = Lambda(self._knn_graph, name='construct_knn_graph')([x])
        # G = tf.numpy_function(self._knn_graph, [x], tf.float32)

        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = Dense(units=dims[i + 1], activation='relu', name='encoder_%d' % i)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            # x = Softmax()(x)

        latent_dim = dims[-1]

        # --- Custom Latent Space Layer
        z_mean = Dense(units=latent_dim, name='Z-Mean')(x)  # Mean component
        z_log_sigma = Dense(units=latent_dim, name='Z-Log-Sigma')(x)  # Standard deviation component
        z = Lambda(self._sampling, name='Z-Sampling-Layer')([z_mean, z_log_sigma, latent_dim])  # Z sampling layer

        # hidden layer
        encoded = Dense(units=latent_dim, name='encoder_%d' % (n_stacks - 1))(z)  # hidden layer, features are extracted from here

        # STEP 2: CREATE DECODER
        x = encoded
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation='relu', name='decoder_%d' % i)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            # x = Softmax()(x)

        # output
        x = Dense(dims[0], name='decoder_0')(x)
        decoded = x

        # STEP 3: CREATE MODELS
        decoder = Model(inputs=encoded, outputs=decoded, name='decoder')
        encoder = Model(inputs=input_visible, outputs=[z_mean, z_log_sigma, z], name='encoder')
        autoencoder = Model(inputs=input_visible, outputs=decoded, name='AE')

        # STEP 4: CREATE CUSTOM LOSS
        z_mean, z_log_sigma, z = encoder(input_visible)

        output = decoder(z)
        # output = autoencoder(visible)

        # Reconstruction loss compares inputs and outputs and tries to minimise the difference
        r_loss = losses.mean_squared_error(input_visible, output)  # use MSE
        r_loss *= dims[0]

        # knn graph
        # g_loss = 1 / K.sum(K.square(z), axis=1)  # norm
        # z_mean = K.softmax(z_mean)
        # zT = K.transpose(K.square(z_mean))

        # x_ = K.expand_dims(z, 0)
        # y_ = K.expand_dims(z, 1)
        # s = x_ - y_
        # distances = tf.norm(s, axis=[-1, 0])
        # D = Lambda(pairwise_distances)([z_mean, z_mean])
        # d_loss = K.mean(K.abs(distances - K.abs(1-K.eye(200)))) # K.var(K.abs(1 - K.sum(D, axis=1)))
        # g_loss *= dims[0]
        #
        # g_loss = K.sum(S, axis=1)

        # KL divergence loss compares the encoded latent distribution Z with standard Normal distribution and penalizes if it's too different
        kl_loss = -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=1)

        # The VAE loss is a combination of reconstruction loss and KL loss
        vae_loss = K.mean(r_loss + kl_loss)  # + nmf_reconstruction_loss)

        # Add loss to the model and compile it
        autoencoder.add_loss(vae_loss)

        return autoencoder, encoder, decoder
