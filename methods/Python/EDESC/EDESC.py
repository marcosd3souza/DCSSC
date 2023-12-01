from __future__ import print_function, division
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from .utils import LoadDataset
import tensorflow as tf  
import keras.backend as K
import warnings
from .AutoEncoder import AE
from .InitializeD import Initialization_D
from .Constraint import D_constraint1, D_constraint2
import time
warnings.filterwarnings("ignore")

   
class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 d,
                 eta,
                 n_clusters,
                 num_sample):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.d = d
        self.eta = eta

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)	

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = self.d
        s = None
        eta = self.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss

		
def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model):

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))


def train_EDESC(X, lr, n_clusters, d, n_z, eta, beta):
    device = 'cpu'
    model = EDESC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=X.shape[1],
        n_z=n_z,
        d=d,
        eta=eta,
        n_clusters=n_clusters,
        num_sample=X.shape[0]).to(device)
    start = time.time()      

    # Load pre-trained model
    # model.pretrain('reuters.pkl')
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Cluster parameter initiate
    data = X
    # y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)

    # Get clusters from Consine K-means 
    # ~ X = hidden.data.cpu().numpy()
    # ~ length = np.sqrt((X**2).sum(axis=1))[:,None]
    # ~ X = X / length
    # ~ y_pred = kmeans.fit_predict(X)
 
    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    # print("Initial Cluster Centers: ", y_pred)
    
    # Initialize D
    D = Initialization_D(hidden, y_pred, n_clusters, d)
    D = torch.tensor(D).to(torch.float32)
    # accmax = 0
    # nmimax = 0
    # y_pred_last = y_pred
    model.D.data = D.to(device)
    
    model.train()
    
    for epoch in range(30):
        x_bar, s, Z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        # y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        # delta_label = np.sum(y_pred != y_pred_last).astype(
        #     np.float32) / y_pred.shape[0]
        # y_pred_last = y_pred
        # # acc = cluster_acc(y, y_pred)
        # nmi = nmi_score(y, y_pred)
        # # if acc > accmax:
        # #     accmax = acc
        # if nmi > nmimax:
        #     nmimax = nmi
        # print('Iter {}'.format(epoch), ', Current nmi {:.4f}'.format(nmi), ':Max nmi {:.4f}'.format(nmimax))
        

        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, Z, pred=s, target=s_tilde, dim=d, n_clusters = n_clusters, beta = beta)

        print(f'epoch: {epoch} loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end-start)
    return Z


def EDESC_exec(X):
    lr = 0.001
    n_clusters = 4
    d = 5
    n_z = 20
    eta = 5
    beta = 0.1

    Z = train_EDESC(X, lr, n_clusters, d, n_z, eta, beta)

    return Z

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=5, type=int)
    #parser.add_argument('--batch_size', default=512, type=int)    
    parser.add_argument('--dataset', type=str, default='reuters')
    parser.add_argument('--pretrain_path', type=str, default='data/reuters')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.dataset = 'reuters'
    if args.dataset == 'reuters':
        args.pretrain_path = 'data/reuters.pkl'
        args.n_clusters = 4
        args.n_input = 2000
        args.num_sample = 10000
        dataset = LoadDataset(args.dataset)   
    print(args)

    Z = train_EDESC()

    # bestacc = 0
    # bestnmi = 0
    # for i in range(10):
    #     acc, nmi = train_EDESC()
    #     # if acc > bestacc:
    #     #     bestacc = acc
    #     if nmi > bestnmi:
    #         bestnmi = nmi
    # print(' Best NMI {:4f}'.format(bestnmi))
