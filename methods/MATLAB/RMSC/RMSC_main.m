function [Z, label] = RMSC_main(fea, numClust, knn0, lambda, metric)
% fea - a cell of feature, size(fea) = [numSample, numFeature]
% numClust - number of desired clusters
% knn0 - number of k-nearest neighbors
% metric - the metric for computing distance, can be (squaredeuclidean,cosine,original)
if nargin < 5
    metric = 'squaredeuclidean';
end
projev = 1.5; 

WW = make_distance_matrix_single(fea, metric);
v = length(fea);
knn = knn0 + 1;
n = length(WW);

% Construct kernel and transition matrix
K = cell(v, 1);
T = cell(v, 1);

% multi-view
%for i = 1:v
%    K{i} = make_kNN_dist(WW{i}, knn);
%    D = spdiags(1 ./ sum(K{i},2), 0, n, n);
%    T{i} = D * K{i};
%end

% single-view
K = make_kNN_dist(WW, knn);
D = spdiags(1 ./ sum(K,2), 0, n, n);
T = [D * K];

clear WW K

opts.DEBUG=0;
opts.eps=1e-7;
opts.max_iter=100;

P_hat = RMSC_sparse2_saveMem(T, lambda, opts);
U_n = baseline_spectral_onRW2(P_hat,numClust,projev);
if any(isnan(U_n))
    U_n(isnan(U_n)) = 0;
end
Z = P_hat
label = kmeans(U_n, numClust,'Replicates',10,'MaxIter',1000);
