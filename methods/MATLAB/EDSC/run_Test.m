% Face clustering on Yale face extended B dataset Using EDSC
% Pan Ji, pan.ji@anu.edu.au
% Oct 2013, ANU
% Modified in Jul 2014
clear all, close all
warning off
%load YaleBCrop025.mat
%data = dlmread('./pixraw10P.csv',';',2,0);
X = rand(1024, 2000);

X = X/max(X(:));		
[D,N] = size(X);                  

% You may tune the lambdas below for better results.
lambda(1) = 0.06;
lambda(2) = 0.01;		
affine = false; outlier = true; Dim = 10; alpha = 4;			
[Z] = edsc(X,5,lambda,affine,outlier,Dim,alpha);

%{
nSet = [5]; %3 5, 8 10];
for i = 1:length(nSet)	
    n = nSet(i);
    idx = Ind{n};
    for j = 1:size(idx,1) %test			
        X = [];
          for p = 1:n
            X = [X Y(:,:,idx(j,p))];
          end
    		X = X/max(X(:));		
        [D,N] = size(X);                  
        
        % You may tune the lambdas below for better results.
        lambda(1) = 0.06;
        lambda(2) = 0.01;		
        affine = false; outlier = true; Dim = 10; alpha = 4;			
        [Z] = edsc(X,s{n},lambda,affine,outlier,Dim,alpha);
            
        %missrateTot{n}(j) = missrate;		
        
        %disp([num2str(n) ' subjects, ' 'sequence ' num2str(j) ': ' num2str(100*missrateTot{n}(j)) '%']);		
    end	
  %  avgmissrate(n) = mean(missrateTot{n});
  %  medmissrate(n) = median(missrateTot{n});	
	%disp([num2str(n) ' subjects: ']);
	%disp(['Mean: ' num2str(100*avgmissrate(n)) '%, ' 'Median: ' num2str(100*medmissrate(n)) '%']);
end
%}
imshow(Z)
