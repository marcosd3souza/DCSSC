import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the Encoder network
class Layer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
        super(Encoder, self).__init__()
        # First encoding layer using EncoderLayer
        self.encoder_layer = Layer(input_dim, hidden_dim, dropout_rate)

        # Latent space transformations
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log-variance of the latent space

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.01 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Pass through the first encoding layer
        h1 = self.encoder_layer(x)

        # Compute the mean and log-variance for the latent space
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)

        z = self.reparameterize(mu, logvar)

        return mu, logvar, z


# Define the Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(Decoder, self).__init__()
        # First decoding layer using DecoderLayer
        self.decoder_layer = Layer(latent_dim, hidden_dim, dropout_rate)

        # Final output layer
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Pass through the decoding layer
        h1 = self.decoder_layer(z)

        # Apply sigmoid activation to the output layer
        return torch.sigmoid(self.fc_output(h1))


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        return self.decoder(z), mu, logvar


# Define the VAE loss function including the contrastive loss
def cvae_loss(recon_A, A, mu, logvar, encoder, margin=1.0):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_A, A, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Contrastive loss
    contrastive = contrastive_loss(encoder, A, margin)

    # #normalize losses
    # losses = torch.tensor([recon_loss, kl_loss, contrastive])
    # recon_loss = (recon_loss - torch.min(losses)) / (torch.max(losses) - torch.min(losses))
    # kl_loss = (kl_loss - torch.min(losses)) / (torch.max(losses) - torch.min(losses))
    # contrastive = (contrastive - torch.min(losses)) / (torch.max(losses) - torch.min(losses))

    # Total loss
    return recon_loss + kl_loss + contrastive


# Contrastive loss function
def contrastive_loss(encoder, A, margin=1.0):
    # positive_pairs = torch.nonzero(A)
    A_inverse = 1 - A
    # negative_pairs = torch.nonzero(A_inverse)

    _, _, z1 = encoder(A)
    _, _, z2 = encoder(A_inverse)

    # Normalize the embeddings
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Compute the positive and negative pairs
    positive_pairs = torch.sum((z1 - z2) ** 2, dim=1)
    negative_pairs = torch.sum((z1 - (-z2)) ** 2, dim=1)

    # Contrastive loss
    contrastive_loss = torch.sum(torch.relu(positive_pairs - margin) + negative_pairs)

    return contrastive_loss

    # loss = 0.0
    # for pos_idx, neg_idx in zip(positive_pairs, negative_pairs):
    #     z_pos = mu[pos_idx[0], pos_idx[1]]
    #     z_neg = mu_inverse[neg_idx[0], neg_idx[1]]
    #
    #     pos_loss = F.mse_loss(z_pos, z_pos)
    #     neg_loss = torch.clamp(margin - F.mse_loss(z_pos, z_neg), min=0.0)
    #
    #     loss += pos_loss + neg_loss
    #
    # return loss / (len(positive_pairs) + len(negative_pairs))


def test():
    # Hyperparameters
    input_dim = 16  # Adjust based on your adjacency matrix size
    hidden_dim = 32
    latent_dim = 5
    learning_rate = 1e-3
    num_epochs = 100

    # Initialize VAE model, optimizer, and data (dummy data used for illustration)
    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Dummy adjacency matrix data for illustration
    A = torch.randint(0, 2, (input_dim, input_dim)).float()

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        optimizer.zero_grad()

        recon_batch, mu, logvar = vae(A)
        loss = vae_loss(recon_batch, A, mu, logvar, vae, A)

        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# test()
