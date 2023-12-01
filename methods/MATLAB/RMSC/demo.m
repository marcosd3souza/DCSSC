pkg load statistics;

X = [rand(1000,4),];
numClust = 4;
knn0 = 5;
lambda = 1;
metric = 'squaredeuclidean';

[Z, label] = RMSC_main(X, numClust, knn0, lambda, metric);

imshow(Z);
