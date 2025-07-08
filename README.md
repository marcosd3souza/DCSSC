# Deep Contrastive Variational Subspace Clustering

![Workflow](https://github.com/marcosd3souza/DCSSC/blob/main/DCSSC.png)


# Abstract

Subspace clustering aims to group data points that lie in multiple low-dimensional subspaces within a high-dimensional space. Traditional shallow methods rely on linear optimization and self-expressive properties but struggle with complex, high-dimensional, and limited data samples. In contrast, Deep subspace clustering methods leverage neural networks to learn feature representations, but often specialize in specific domains, such as images, and require extensive training datasets. Furthermore, most approaches rely solely on raw features, making it difficult to capture meaningful structures in data with few examples. To address these challenges, we propose Deep Contrastive and Variational Subspace Spectral Clustering (DCSSC), a novel method that integrates relational information through a similarity matrix, enhancing clustering performance. DCSSC consists of three stages: (1) constructing a neighborhood graph for data representation, (2) training a contrastive variational autoencoder (cVAE) to learn embeddings, and (3) generating an enhanced distance matrix for spectral clustering. Our method is computationally efficient and performs well with minimal training data. Extensive experiments on simulated and 19 real-world datasets from diverse domains, including text, images, and bioinformatics, demonstrate the superiority of DCSSC over state-of-the-art deep and shallow clustering methods. Code is available at https://github.com/marcosd3souza/DCSSC.

# Installation

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

Run main.py
``` 
python main.py
```
# Citation

```
@article{DESOUZAOLIVEIRA2025130901,
title = {Deep contrastive variational subspace clustering},
journal = {Neurocomputing},
pages = {130901},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.130901},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225015735},
author = {Marcos {de Souza Oliveira} and Sergio Ricardo {de Melo Queiroz} and Cleber Zanchettin and Francisco {de Assis Tenório de Carvalho}},
keywords = {Deep subspace clustering, Contrastive learning, Variational autoencoder, Neighborhood graph, Spectral clustering}
}
```
# DOI
https://doi.org/10.1016/j.neucom.2025.130901
