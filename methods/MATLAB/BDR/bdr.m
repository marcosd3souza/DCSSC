function [Z] = bdr(X)

lambda = 50
gamma = 1
rho = 0.4
r = 0

X = DataProjection(X,r);

for l = 1 : size(X,2)
    X(:,l) = X(:,l)/norm(X(:,l));
end

[Z] = BDR_solver(X,lambda,gamma);
%Z = BuildAdjacency(thrC(Z,rho));