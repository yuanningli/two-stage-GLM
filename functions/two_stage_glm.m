function [pred_post, pred_pre, W_post, W_pre, load_post, load_pre, MI] ...
    = two_stage_glm(train_data, train_tags, test_data, feat_index, groups)
% Fitting the two-stage GLM model
% Inputs:
%  train_data - N x P training data matrix
%  train_tags - N dimensional vector for training data labels
%  test_data - M x P testing data matrix
%  feat_index - P dimensional vector, indicating p1 pre-stim(1) and 
%               p2 post-stim(2) features
%  groups - p1 dimensional vector of group index indicating the group 
%           structure in group-lasso penalty
% Outputs:
%  predict_post - M dimensional vector, predicted labels of the test set 
%                 using only post-stim features
%  predict_pre - M dimensional vector, predicted labels of the test set 
%                 using both post-stim features and pre-stim modulations
%  W_post - p1 dimensional vector weights for the post-stim features
%  W_pre - p2 dimensinal weights for the pre-stim features
%  load_post - M dimensional vector, loadings of the post-stim features
%  load_pre - M dimensional vector, loadings of all features in full model
%  MI - M dimensional vector, modulation index for each trial
%
% Yuanning Li (yuanningli@gmail.com), 2020
%


% standarlize the data by z-scoring
gmean = mean(train_data,1);
gstd = std(train_data,0,1);
train_data_n = train_data - repmat(gmean,size(train_data,1),1);
train_data_n = train_data_n ./ repmat(gstd,size(train_data,1),1);
test_data_n = test_data - repmat(gmean,size(test_data,1),1);
test_data_n = test_data_n ./ repmat(gstd,size(test_data,1),1);

% find pre and post feature groups
idx_pre = find(feat_index == 1);
idx_post = find(feat_index == 2);

% STEP1: evaluate beta weights for post features only
X = train_data_n(:,idx_post);
csvwrite('./functions/sglX.dat',X)
csvwrite('./functions/sgly.dat',round(train_tags))
groupid = (1 : size(X,2))';
csvwrite('./functions/sglgroup.dat',groupid);
system('Rscript ./functions/grpfunc.R')
W_post = csvread('./functions/grpW.csv',1,1);
beta_post = W_post(2:end);
alpha_post = W_post(1);

% fix the post-stim part
offset_post_train = train_data_n(:,idx_post) * beta_post + alpha_post;
offset_post_test = test_data_n(:,idx_post) * beta_post + alpha_post;

% post-stim predictions
pred_prob_post = 1 ./ (1 + exp(- offset_post_test));
pred_post = double(pred_prob_post >= 0.5);

% include pre-stim features
X_pre = [offset_post_train, train_data_n(:,idx_pre)];
X_pre_test = [offset_post_test, test_data_n(:,idx_pre)];
groupid_pre = [0; (groups)'];

% STEP2: evaluate beta weights for pre features
csvwrite('./functions/sglX.dat',X_pre);
csvwrite('./functions/sglgroup.dat',groupid_pre);
system('Rscript ./functions/grpfunc.R')
W_pre = csvread('./functions/grpW.csv',1,1);
beta_pre = W_pre(2:end);
alpha_pre = W_pre(1);

% full model predictions
pred_prob_pre = 1 ./ (1 + exp(- (X_pre_test * beta_pre + alpha_pre)));
pred_pre = double(pred_prob_pre >= 0.5);

% loadings 
load_post = offset_post_test;
load_pre = X_pre_test * beta_pre + alpha_pre;
MI = (test_data_n(:,idx_pre) * beta_pre(2:end) + alpha_pre);










