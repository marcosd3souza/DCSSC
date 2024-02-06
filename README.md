# DGSSC
Deep Generative Subspace Spectral Clustering based on Low-Rank Factorization and Variational Autoencoder

### Install miniconda #################

> mkdir -p ~/miniconda3
 
>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

>bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

>rm -rf ~/miniconda3/miniconda.sh

#######################################
### Create conda env and install packages ###
```
conda create -name subspace-clustering
conda activate subspace-clustering
conda install pip
```

```
pip install oct2py cvxpy pandas opencv-python-headless torchvision munkres
```

Obs1: there's a conflict between oct2py and opencv-python.
<br>Obs2: Please install opencv-python-headless
<br>Obs3: opencv is necessary to convert CIFAR10 to grayscale

```
conda install -c conda-forge tensorflow keras scikit-learn
```

``` 
conda install -c pytorch pytorch
```

#######################################
