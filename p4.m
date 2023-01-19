clear;
clc;
close all;
format shortEng;
rng("shuffle")

% Read the datasets
learning_set = dlmread("instantaneous.txt");
test_set = dlmread("instantaneous_test.txt");

% Shuffling
data_set = [learning_set;test_set];
data_set(:,101)=data_set(:,101);
data_set = data_set(randperm(size(data_set, 1)), :);
[trainId,valId,testId] = dividerand(size(data_set,1),0.5,0,0.5);

% Use Indices to parition the matrix
learning_set = data_set(trainId,:);
test_set = data_set(testId,:);

% Get the labels for each set
learning_descriptors = learning_set(:,1:100);
learning_labels = learning_set(:,101)+1;
test_descriptors = test_set(:,1:100);
test_labels = test_set(:,101)+1;

% Principle component analysis
d_p = 100;
[learning_descriptors, targets, UW, m, W] = PCA(learning_descriptors', learning_labels', d_p);
learning_descriptors = learning_descriptors';
test_descriptors = (W*test_descriptors')';

% Apply one-vs-all SVM for each class
X = learning_descriptors;
for i = 1:5
    Y = (learning_labels == i);
    models{i} =fitcsvm(X,Y,'Standardize',true,'KernelFunction','linear');
end

% Calculate the prediction score for each model-sample pair
for i =1:5
    [~,scores{i}]=predict(models{i},test_descriptors);
end

% Find the center of the classes
for i=1:5
    c(i,:) = mean(learning_descriptors(learning_labels==i,:));
end

% Calculate the pairwise class center seperations
for i =1:5
    for j =1:5
        class_dist(i,j) = norm(c(i,:)-c(j,:));
    end
end
[B,I] = sort(sum(class_dist,2),'descend');

% The resulting decision tree with prioritized SVMs
test_preds = zeros(length(test_labels),1);
for i=1:length(test_labels)
    if scores{I(1)}(i,2) > 0  
        test_preds(i,1) = 4; 
    else
        if scores{I(2)}(i,2) > 0
            test_preds(i,1) = 3; 
        else
            if scores{I(3)}(i,2) > 0
                test_preds(i,1) = 5;
            else 
                if scores{I(4)}(i,2) > 0
                    test_preds(i,1) = 1;
                else
                    if scores{I(5)}(i,2) > 0
                        test_preds(i,1) = 2;
                    else
                        [M,max_ind] = max([scores{1}(i,2), ...
                            scores{2}(i,2), ...
                            scores{3}(i,2), ...
                            scores{4}(i,2), ...
                            scores{5}(i,2)]);
                        test_preds(i,1) = max_ind;
                    end
                end
            end
        end
    end
end

test_cm = confusionmat(test_labels,test_preds);
accuracy=sum(diag(test_cm))/sum(test_cm,"all");
figure;
confusionchart(test_labels,test_preds, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');





