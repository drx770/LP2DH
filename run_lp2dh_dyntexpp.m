clc;
clear;
close all;

addpath('./third_party/wen_yin');

%% DynTex++ path
data_root = './DynTex++'; % Put your path here

%% Parameters
P = 3;
M = 16;
C = 3000;
K = 10;
lambda1 = 10;
lambda2 = 1;
lambda3 = 1000;
seed = 2026308;

%% Dataset and split
load(fullfile(data_root,'dyntex++_info.mat'),'imagemaster');
load('split.mat','split');

%% W and dictionary
load('W.mat','W');
load('dictionary.mat','dictionary');

% % To train W and the dictionary, comment the two load commands above and use:
% [W,dictionary] = lp2dh_train(imagemaster,data_root,split.train_idx, ...
%     P,M,C,K,lambda1,lambda2,lambda3,seed);

%% Video features
load('features.mat','feat_train','feat_test','y_train','y_test');

% % To recompute all video features, comment the load command above and use:
% hist_all = lp2dh_encode_features(imagemaster,data_root,W,dictionary,P);
% [feat_train,feat_test,y_train,y_test] = lp2dh_histogram_pca(hist_all,split);

%% Cosine 1-NN
distance = pdist2(feat_test,feat_train,'cosine');
distance(~isfinite(distance)) = inf;
[~,nearest] = min(distance,[],2);
y_pred = y_train(nearest);
accuracy = mean(y_pred==y_test);
correct = sum(y_pred==y_test);

fprintf('Accuracy: %.4f%% (%d / %d correct)\n', ...
    100*accuracy,correct,numel(y_test));
