function [F,info] = lp2dh_build_lle(X,K,rng_reset)

rng(rng_reset,'twister'); 
N = size(X,1);
regularization = 1e-3;
absolute_jitter = 1e-8;

searcher = createns(X,'NSMethod','kdtree','Distance','euclidean','BucketSize',100);
raw = knnsearch(searcher,X,'K',K+1);
for i = find(raw(:,1)~=(1:N)')'
    row = raw(i,:);
    row(row==i) = [];
    if numel(row)<K
        extra = setdiff(1:N,[i row],'stable');
        row = [row extra(1:K-numel(row))];
    end
    raw(i,2:K+1) = row(1:K);
end
neighbors = raw(:,2:K+1);

weights = zeros(N,K);
one = ones(K,1);
for i = 1:N
    Z = X(neighbors(i,:),:)-X(i,:);
    G = Z*Z';
    jitter = regularization*trace(G)/K+absolute_jitter;
    G = (G+G')/2+jitter*eye(K);
    w = G\one;
    denominator = sum(w);
    if ~all(isfinite(w)) || ~isfinite(denominator) || abs(denominator)<eps
        w = one/K;
    else
        w = w/denominator;
    end
    weights(i,:) = w';
end

rows = reshape(neighbors',[],1);
cols = repelem((1:N)',K);
A = sparse(rows,cols,reshape(weights',[],1),N,N);
residual = X-A'*X;
F = residual'*residual/N;
F = (F+F')/2;
info = struct('K',K,'num_points',N,'neighbors',neighbors,'weights',weights, ...
    'orientation','A(neighbor,target)=weight');
end
