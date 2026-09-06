function [W,dictionary,info] = lp2dh_train(imagemaster,data_root,train_idx, ...
    P,M,C,K,lambda1,lambda2,lambda3,seed)
%LP2DH_TRAIN Historical shared-pool LP2DH training and dictionary learning.

train_idx = sort(train_idx(:));
pool_count = 20000;
pool_seed = seed + 200000;       % 2226308 for the reproduced trial
lle_seed = seed + 1;
dictionary_seed = seed + 100000;

cache_dir = 'cache';
if ~isfolder(cache_dir), mkdir(cache_dir); end
pool_file = fullfile(cache_dir,'pdv_pool.mat');
pool = build_shared_pool(imagemaster,data_root,train_idx,P,pool_count,pool_seed,pool_file);

% The same pool supplies hash statistics, LLE samples, and dictionary codes.
base = stream_stats(pool,[],M);
[V,E] = eig((base.xtx+base.xtx')/2,'vector');
[~,order] = sort(E,'descend');
[W,~] = qr(V(:,order(1:M)),0);

X_lle = sample_pool(pool,100000,lle_seed);
[F,lle_info] = lp2dh_build_lle(X_lle,K,seed+2);

Q = F + lambda1*base.xtx/base.N ...
    + lambda2*(base.sum_x*base.sum_x')/base.N ...
    - lambda3*(base.xtx/base.N-base.mean_x*base.mean_x');
Q = (Q+Q')/2;

history = [];
previous_counts = [];
for outer = 1:20
    stats = stream_stats(pool,W,M);
    linear = -2*lambda1*stats.xtb/base.N ...
        - lambda2*(base.sum_x*ones(1,M));
    W_old = W;
    [W,optimizer_info] = lp2dh_optimize_W(W,Q,linear);
    relative_W_change = norm(W-W_old,'fro')/max(1,norm(W_old,'fro'));
    if isempty(previous_counts)
        code_distribution_change = NaN;
    else
        code_distribution_change = sum(abs(double(stats.code_counts)- ...
            double(previous_counts)))/(2*base.N);
    end
    history = [history; struct('iteration',outer, ...
        'relative_W_change',relative_W_change, ...
        'code_distribution_change',code_distribution_change, ...
        'orthogonality_error',norm(W'*W-eye(M),'fro'), ...
        'inner_iterations',optimizer_info.itr)]; %#ok<AGROW>
    previous_counts = stats.code_counts;
    if outer>=2 && relative_W_change<1e-5 && code_distribution_change<1e-5
        break;
    end
end

final_stats = stream_stats(pool,W,M);
dictionary = learn_dictionary(final_stats.code_counts,M,C,dictionary_seed);
info = struct('pool_file',pool.file,'pool_seed',pool_seed, ...
    'lle_seed',lle_seed,'dictionary_seed',dictionary_seed, ...
    'shared_pool',true,'lle',lle_info,'history',history, ...
    'orthogonality_error',norm(W'*W-eye(M),'fro'));
end

function pool = build_shared_pool(imagemaster,data_root,train_idx,P,count,seed,file)
expected = numel(train_idx)*count;
if isfile(file)
    saved = load(file,'metadata');
    metadata = saved.metadata;
    pool = struct('file',file,'rows',expected,'dimension',26, ...
        'count_per_video',count,'metadata',metadata);
    return;
end

m = matfile(file,'Writable',true);
m.X(expected,26) = int16(0);
rng(seed,'twister');
sampled_center_indices = zeros(numel(train_idx),count,'uint32');
for q = 1:numel(train_idx)
    video_index = train_idx(q);
    video = load(fullfile(data_root,imagemaster{video_index}.filepath),'subv');
    valid_count = prod(size(video.subv)-2);
    centers = sort(randperm(valid_count,count));
    sampled_center_indices(q,:) = uint32(centers);
    X = lp2dh_fibonacci_transform(lp2dh_extract_pdv(video.subv,P,centers));
    rows = (q-1)*count + (1:count);
    m.X(rows,:) = int16(X);
    if mod(q,50)==0
        fprintf('Shared PDV pool: %d/%d training videos\n',q,numel(train_idx));
    end
end
metadata = struct('complete',true,'seed',seed,'P',P, ...
    'count_per_video',count,'train_idx',train_idx, ...
    'sampled_center_indices',sampled_center_indices);
save(file,'metadata','-append');
pool = struct('file',file,'rows',expected,'dimension',26, ...
    'count_per_video',count,'metadata',metadata);
end

function X = sample_pool(pool,n,seed)
rng(seed,'twister');
indices = sort(randperm(pool.rows,n));
m = matfile(pool.file);
X = zeros(n,pool.dimension);
block = 250000;
write = 1;
for first = 1:block:pool.rows
    last = min(pool.rows,first+block-1);
    selected = indices>=first & indices<=last;
    count = nnz(selected);
    if count==0, continue; end
    source = double(m.X(first:last,:));
    local = indices(selected)-first+1;
    X(write:write+count-1,:) = source(local,:);
    write = write+count;
end
end

function stats = stream_stats(pool,W,M)
m = matfile(pool.file);
N = pool.rows;
D = pool.dimension;
stats = struct('N',N,'sum_x',zeros(D,1),'xtx',zeros(D,D), ...
    'xtb',[],'code_counts',[]);
if ~isempty(W)
    stats.xtb = zeros(D,M);
    stats.code_counts = zeros(2^M,1,'uint64');
    powers = 2.^(0:M-1);
end
for first = 1:250000:N
    last = min(N,first+250000-1);
    X = double(m.X(first:last,:));
    stats.sum_x = stats.sum_x+sum(X,1)';
    stats.xtx = stats.xtx+X'*X;
    if ~isempty(W)
        B = double(X*W>=0);
        stats.xtb = stats.xtb+X'*B;
        ids = 1+uint32(B*powers');
        stats.code_counts = stats.code_counts + ...
            uint64(accumarray(double(ids),1,[2^M 1]));
    end
end
stats.mean_x = stats.sum_x/N;
end

function dictionary = learn_dictionary(counts,M,C,seed)
counts = double(counts(:));
observed = find(counts>0);
frequency = counts(observed);
ids = uint32(observed-1);
codes = zeros(numel(ids),M);
for bit = 1:M, codes(:,bit)=bitget(ids,bit); end

rng(seed,'twister');
sampled = randsample(numel(observed),120000,true,frequency/sum(frequency));
rng(seed,'twister');
options = statset('MaxIter',100,'Display','off');
[~,dictionary] = kmeans(codes(sampled,:),C,'Distance','sqeuclidean', ...
    'Start','plus','Replicates',3,'EmptyAction','singleton','Options',options);
end
