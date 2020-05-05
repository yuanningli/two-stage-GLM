function [dp, TP, FP] = eval_pred(pred, test_tags, targetcode)
% evaluate classifier predictions
% Inputs: 
%   pred - vector of predicted labels
%   test_tags - vector of true labels for the test set
%   targetcode - the label code for the target category
% Outputs:
%   dp - d' value
%   TP - true positive rate for target category
%   FP - false positive rate for target category
% 
  
unique_tags = unique(test_tags);
for i = 1 : length(unique_tags)
    test_id_count(i) = length(find(test_tags == unique_tags(i)));
end
test_target_idx = find(test_tags == targetcode);
test_nontarget_idx = find(test_tags ~= targetcode);

acc = 1 - length(find(test_tags - pred)~=0) / length(test_tags);
TPR_target = length(find(pred(test_target_idx) == targetcode)) / length(test_target_idx);
FPR_target = length(find(pred(test_nontarget_idx) == targetcode)) / (length(test_tags) - length(test_target_idx));
TP = min(TPR_target, (test_id_count(targetcode)-1)/test_id_count(targetcode));
TP = max(TP, 1/test_id_count(targetcode));
FP = max(FPR_target, 1/(sum(test_id_count) - test_id_count(targetcode)));
FP = min(FP, 1 - 1/(sum(test_id_count) - test_id_count(targetcode)));
dp = norminv(TP) - norminv(FP);

