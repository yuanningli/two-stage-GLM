clear all

load ./data/data.mat
load ./data/time_window.mat

%% assign pre-stim and post-stim features, 1 for pre-stim, 2 for post-stim
feat_idx_BHA = zeros(1,size(data_BHA,2));
feat_idx_BHA(time_win_BHA_pre) = 1;
feat_idx_BHA(time_win_BHA_post) = 2;
feat_idx_stP = zeros(1,size(data_stP,2));
feat_idx_stP(time_win_stP_pre) = 1;
feat_idx_stP(time_win_stP_post) = 2;

feat_idx_stP_BHA = cat(2,feat_idx_BHA,feat_idx_stP);
all_data_stP_BHA = cat(2,data_BHA,data_stP);

all_data_phase_group = zeros(size(data_phase,1),2*size(data_phase,2));
all_data_phase_group(:,1:2:end) = sin(data_phase);
all_data_phase_group(:,2:2:end) = cos(data_phase);

all_data = cat(2,all_data_stP_BHA,all_data_phase_group);
feat_idx = cat(2,feat_idx_stP_BHA, ones(1,size(all_data_phase_group,2)));

%% assign group index for group-lasso penalty
group_phase = zeros(1,size(all_data_phase_group,2));
group_phase(1:2:end) = 1:1:size(data_phase,2);
group_phase(2:2:end) = 1:1:size(data_phase,2);
group_phase = group_phase + length(time_win_BHA_pre)+length(time_win_stP_pre);
groups = [1:(length(time_win_BHA_pre)+length(time_win_stP_pre)),group_phase];

%% cross-validation
N = length(all_data_tags);
num_cv = 5;
cv = cvpartition(N, 'k', num_cv);

dprime_post = zeros(num_cv);
dprime_pre = zeros(num_cv);

all_MI = zeros(N);

for i = 1 : num_cv
    train_idx = cv.training(i);
    test_idx = cv.test(i);
    
    train_data = all_data(train_idx,:);
    train_tags = all_data_tags(train_idx);
    test_data = all_data(test_idx,:);
    test_tags = all_data_tags(test_idx);
    
    [pred_post, pred_pre, W_post, W_pre, loading_post, loading_pre, MI] ...
    = two_stage_glm(train_data, train_tags, test_data, feat_idx, groups);
    
    dprime_post(i) = eval_pred(pred_post, test_tags, 1);
    dprime_pre(i) = eval_pred(pred_pre, test_tags, 1);
    
    all_MI(test_idx) = MI;
end
