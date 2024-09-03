import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Learnable weights
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # Linear transformation
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked attention (only consider neighbors)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention  # Return attention coefficients
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # Number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout, alpha, n_heads):
        super(GAT, self).__init__()

        # Multi-head GAT layer
        self.attentions = nn.ModuleList(
            [GATLayer(n_features, n_hidden, dropout, alpha, concat=True) for _ in range(n_heads)])
        self.out_att = GATLayer(n_hidden * n_heads, n_classes, dropout, alpha, concat=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.dropout(x)

        attentions_list = []
        x_cat = []
        for att in self.attentions:
            out, attention = att(x, adj)
            x_cat.append(out)
            attentions_list.append(attention)

        # Concatenate the results of all attention heads
        x = torch.cat(x_cat, dim=1)
        x = self.dropout(x)
        x, final_attention = self.out_att(x, adj)

        # Average attention across heads
        avg_attention = torch.mean(torch.stack(attentions_list), dim=0)

        return F.log_softmax(x, dim=1), avg_attention


# Example usage:
def test():
    # Number of nodes and features
    N = 5  # Number of nodes
    F_in = 10  # Number of input features
    F_out = 2  # Number of output classes

    # Input features (N x F_in)
    X = torch.rand(N, F_in)

    # Adjacency matrix (N x N)
    adj = torch.tensor([[1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1]], dtype=torch.float32)

    # Hyperparameters
    n_hidden = 8  # Number of hidden units per head
    n_heads = 8  # Number of attention heads
    dropout = 0.6  # Dropout rate
    alpha = 0.2  # Negative slope for LeakyReLU

    # Initialize the GAT model
    model = GAT(n_features=F_in, n_hidden=n_hidden, n_classes=F_out, dropout=dropout, alpha=alpha, n_heads=n_heads)

    # Forward pass to get the output and attention coefficients
    output, attention_matrix = model(X, adj)

    # Threshold for creating new edges (e.g., 0.5)
    threshold = 0.2
    new_adj_matrix = (attention_matrix > threshold).float()

    print("New Adjacency Matrix based on learned attention coefficients:")
    print(new_adj_matrix)

# test()
