function Z = postProC(C,d,alpha)
%if(nargin<4)
%	alpha = 4;
%end
%if(nargin<3)
%	d = 4;
%end
N = length(C);

C = (C + C')/2;
r = d*10+1;
opts.u0 = ones(N,1);
[U,S,~] = svds(C,r,'L',opts);
S = diag(S);
%U = U*diag(sqrt(S));
U = norm(U);
U = sqrt(sum(U.^2,2));
Z = U*U';
%Z = Z.^alpha;

% ncut
%L = (L + L')/2;
%grp = ncutW(L,K); % get your ncutW funtion via http://www.cis.upenn.edu/~jshi/software/

%s = zeros(N,1);
%for i = 1:K
%    s = s+grp(:,i)*i;
%end

end