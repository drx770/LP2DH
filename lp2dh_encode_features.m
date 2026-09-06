function hist_all = lp2dh_encode_features(imagemaster,data_root,W,dictionary,P)

M = size(W,2);
C = size(dictionary,1);
[lookup,powers] = codeword_lookup(dictionary,M);
hist_all = zeros(numel(imagemaster),C,'uint32');

for video_index = 1:numel(imagemaster)
    video = load(fullfile(data_root,imagemaster{video_index}.filepath),'subv');
    X = lp2dh_fibonacci_transform(lp2dh_extract_pdv(video.subv,P));
    binary_codes = double(X*W>=0);
    code_ids = 1+uint32(binary_codes*powers');
    words = lookup(code_ids);
    hist_all(video_index,:) = uint32(accumarray(double(words),1,[C 1])');
    if mod(video_index,50)==0
        fprintf('Video histograms: %d/%d\n',video_index,numel(imagemaster));
    end
end
end

function [lookup,powers] = codeword_lookup(dictionary,M)
ids = uint32((0:2^M-1)');
codes = zeros(numel(ids),M);
for bit = 1:M, codes(:,bit)=bitget(ids,bit); end
lookup = zeros(numel(ids),1,'uint32');
for first = 1:4096:numel(ids)
    rows = first:min(numel(ids),first+4095);
    lookup(rows) = uint32(knnsearch(dictionary,codes(rows,:), ...
        'K',1,'Distance','euclidean'));
end
powers = 2.^(0:M-1);
end
