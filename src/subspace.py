import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from matplotlib import pyplot as plt

from scipy import sparse
from sklearn.manifold import TSNE
# import umap

import os

from sklearn.neighbors import kneighbors_graph

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import pairwise_distances

# import keras
from keras import backend as K
from keras import losses
from keras.models import Model
from keras.layers import Lambda, Dense, Input, LeakyReLU, ReLU, Dropout, BatchNormalization, Softmax
# from sklearn.neighbors import kneighbors_graph

from src.cVAE import cvae_loss, VAE
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

# import tensorflow as tf
# tf.compat.v1.enable_v2_behavior()


class SubspaceRepresentation:
    def __init__(self, D_input):
        self.D_input = D_input
        self.losses = []
        # self.D_original = D_original
    
    def eig_transform(self, n_vals=10):
        # # # Graph Laplacian.
        L = sparse.csgraph.laplacian(csgraph=self.D_input, normed=True)
        # graph_laplacian = graph_laplacian_s.toarray()

        eigenvals, eigenvcts = np.linalg.eig(L)

        vals = eigenvals.real
        vecs = eigenvcts.real

        positive = np.where(vals>0)

        vals_index = np.argsort(vals[positive])[0:n_vals]
        temp = vecs[:, vals_index]

        X_eig = np.sort(temp, axis=0)

        return pairwise_distances(X_eig)
    
    def cvae_transform(self, y):
        # Hyperparameters
        input_dim = self.D_input.shape[0]  # Adjust based on your adjacency matrix size
        hidden_dim = 100
        latent_dim = 10
        learning_rate = 0.01
        num_epochs = 150

        # losses = []

        A = torch.tensor(self.D_input, dtype=torch.float)

        # Initialize VAE model, optimizer, and data (dummy data used for illustration)
        vae = VAE(input_dim, hidden_dim, latent_dim)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        # Dummy adjacency matrix data for illustration
        # A = torch.randint(0, 2, (input_dim, input_dim)).float()

        # Training loop
        for epoch in tqdm(range(1, num_epochs + 1)):
            vae.train()
            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(A)
            loss = cvae_loss(recon_batch, A, mu, logvar, vae.encoder)

            # if epoch % 20 == 0:
            #     vae.encoder.eval()
            #     with torch.no_grad():
            #         z = vae.encoder(A)[0].numpy()
            #         plt.gca().set_axis_off()
            #         s = kneighbors_graph(z, n_neighbors=50, mode='connectivity').toarray()
            #         plt.imshow(s, cmap='hot')
            #         # proj = TSNE(n_components=2).fit_transform(z)
            #         # plt.scatter(proj[:, 0], proj[:, 1], c=y)
            #         # plt.savefig(f'knn_epoch_{epoch}_loss_{loss}.eps', format='eps', bbox_inches='tight')
            #         # plt.savefig(f'tsne_epoch_{epoch}_loss_{int(np.round(loss, 0))}.eps', format='eps', bbox_inches='tight')
            #         plt.show()

            loss.backward()
            optimizer.step()

            # losses.append(loss.item())
            # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
            # Logging
            if epoch % 10 == 0:
                tqdm.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # plt.plot(losses)
        # plt.savefig(f'losses_knn.eps', format='eps', bbox_inches='tight')
        # plt.savefig(f'losses_tsne.eps', format='eps', bbox_inches='tight')
        # plt.show()

        vae.eval()
        with torch.no_grad():
            return vae.encoder(A)[0].numpy()

    def vae_transform(self):

        dims = [self.D_input.shape[0], 100, 10]

        rng_ep = [1, 10, 50, 100]

        for ep in rng_ep:
            pretrain_epochs = ep#100
            batch_size = 512

            autoencoder, encoder, decoder = self._vae_model(dims)

            autoencoder.compile(optimizer='adam')
            history = autoencoder.fit(
                self.D_input,
                self.D_input,
                batch_size=batch_size,
                epochs=pretrain_epochs,
                verbose=0
            )

            self.losses = history.history

            loss = history.history['loss'][-1]

            D = self.D_input

            Z = encoder.predict(D)[2]
            Zs = np.sort(Z, axis=0)
            autoexp = pairwise_distances(Zs) #Zs.dot(Zs.T)
            # autoexp[autoexp < 0] = 0
            th = 0.4#np.median(autoexp)
            autoexp[autoexp > th] = 1
            autoexp[autoexp < th] = 0
            # autoexp = autoexp * 100

            # proj = TSNE(n_components=2).fit_transform(autoexp)
            # plt.scatter(proj[:, 0], proj[:, 1], c=y)
            # plt.gca().set_axis_off()
            # plt.imshow(autoexp)
            # # plt.imshow(Z)
            # plt.savefig(f'Z_epoch_{ep}_loss_{loss}.eps', format='eps', bbox_inches='tight')
            # plt.show()

        return Z

    def _sampling(self, args):
        z_mean, z_log_sigma, latent_dim = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0., stddev=0.001)
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
            x = Dropout(0.1)(x)

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
            # x = Softmax()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.1)(x)

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
