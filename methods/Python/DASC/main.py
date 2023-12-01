import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.datasets import make_blobs

# import train_mnist
from .train import train


# import utils

def load_data(path='data/COIL20.mat'):
    try:
        print("loading data from {}\{}...".format(os.getcwd(), path))
        coil20 = sio.loadmat(path)
        img = coil20['fea']
        label = coil20['gnd']
        img = np.reshape(img, (img.shape[0], 32, 32, 1))
        return img, label
    except Exception as e:
        print(e)
        return None


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    for i in range(10):
        idx = np.where(y_train == i)[0][0:100]
        if i == 0:
            x = x_train[idx, :, :]
            y = y_train[idx]
        else:
            x = np.concatenate([x, x_train[idx, :, :]], axis=0)
            y = np.concatenate([y, y_train[idx]], axis=0)

    x = np.reshape(x, [-1, 28, 28, 1])

    return x, y


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--save_dir', default="logs/weights.h5")
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--pretrain', type=bool, default=False)

    return parser.parse_args(argv)

def dasc_exec(X_img, n_input, name):
    args = parse_arguments(sys.argv[1:])
    X = X_img

    train_db = tf.data.Dataset.from_tensor_slices((X, X))
    train_db = train_db.batch(X.shape[0]).shuffle(30)

    Z = train(name, train_db, epoch_num=args.epoch, batch_size=X.shape[0], input_shape=n_input, pre_train_epoch=args.pretrain_epoch,
              alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d)

    return Z


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    x, y = make_blobs(n_samples=150, cluster_std=10.0, centers=10, n_features=1024, random_state=0)
    x = x.reshape((-1, 32, 32, 1))
    train_db = tf.data.Dataset.from_tensor_slices((x, x))

    train_db = train_db.batch(150).shuffle(30)

    Z = train('test', train_db, epoch_num=args.epoch, batch_size=args.batch_size, input_shape=[32, 32, 1], pre_train_epoch=args.pretrain_epoch,
              alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d)

# if args.dataset == "mnist":
# 	x, y = load_mnist()
# 	train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 	train_db = train_db.batch(1000).shuffle(30)
# 	theta = train_mnist.train(train_db, epoch_num=args.epoch, batch_size=args.batch_size, pre_train_epoch=args.pretrain_epoch,
# 			  alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d, pretrain=args.pretrain)
#
# elif args.dataset == "coil20":
# 	x, y = load_data(path="data/COIL20.mat")
# 	train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 	train_db = train_db.batch(1440).shuffle(30)
#
# 	theta = train(train_db, epoch_num=args.epoch, batch_size=args.batch_size, pre_train_epoch=args.pretrain_epoch,
# 			  alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d, pretrain=args.pretrain)
