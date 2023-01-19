function [precision, overall_precision, recall, overall_recall] = ROCfromConf(confusion_matrix)
%reference:https://www.youtube.com/watch?v=5mVv2VocH2o
    confusion_matrix_trans=confusion_matrix';
    diagonal = diag(confusion_matrix_trans);
    sum_of_rows = sum(confusion_matrix_trans,2);
    sum_of_cols = sum(confusion_matrix_trans,1);
    
    precision = diagonal ./ sum_of_rows;
    precision(isnan(precision(:,1)))=0;
    overall_precision = mean(precision)
    
    recall = diagonal ./ sum_of_cols';
    recall(isnan(recall(:,1)))=0;
    overall_recall = mean(recall)
end