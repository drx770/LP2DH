function [feat_train,feat_test,y_train,y_test,pca_model] = ...
    lp2dh_histogram_pca(hist_all,split)

hist_normalized = double(hist_all).^0.4;
hist_normalized = hist_normalized ./ ...
    max(sqrt(sum(hist_normalized.^2,2)),eps);

hist_train = hist_normalized(split.train_idx,:);
hist_test = hist_normalized(split.test_idx,:);
[coeff,score,latent,~,explained,mu] = pca(hist_train);
dimension = find(cumsum(explained)>=98,1);

feat_train = score(:,1:dimension);
feat_test = (hist_test-mu)*coeff(:,1:dimension);
scale = sqrt(max(latent(1:dimension)',eps));
feat_train = feat_train./scale;
feat_test = feat_test./scale;
y_train = split.train_labels;
y_test = split.test_labels;
pca_model = struct('mode','variance','variance_retained',0.98, ...
    'dimension',dimension,'whiten',true,'mu',mu, ...
    'coeff',coeff(:,1:dimension),'latent',latent(1:dimension),'scale',scale);
end

