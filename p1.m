clc;
clear;
close all;
format shortEng;
warning('on','all');
rng("shuffle")

% Set number of classes
num_classes = 5;

% Classes
classes = 1:5;

% Load learning and test sets
learning_set = dlmread('instantaneous.txt');
test_set = dlmread('instantaneous_test.txt');

% Shuffling
data_set = [learning_set;test_set];
data_set(:,101)=data_set(:,101)+1;
data_set = data_set(randperm(size(data_set, 1)), :);

[trainId,valId,testId] = dividerand(size(data_set,1),0.5,0,0.5);

% Use Indices to parition the matrix
learning_set = data_set(trainId,:);
val_set = data_set(valId,:);
test_set = data_set(testId,:);

% Extract the descriptors (normalized), sequence IDs and labels from the sets
learning_descriptors = normalize(learning_set(:, 1:100),"range");
learning_labels = learning_set(:, 101);
learning_seq_ids = learning_set(:, 102);
test_descriptors = normalize(test_set(:, 1:100),"range");
test_labels = test_set(:, 101);
test_seq_ids = test_set(:, 102);
[GC,GR,GP]=groupcounts(learning_labels);
priors = GC/sum(GC);

% Initialize vector to store class labels for test data
test_predictions = zeros(size(test_labels, 1), 1);

% PCA
d_p=11;
[learning_descriptors, targets, UW, m, W] = PCA(learning_descriptors', learning_labels', d_p);
learning_descriptors = learning_descriptors';
test_descriptors = (W*test_descriptors')';

% Initialize cell array to store clustering information
cluster_info = cell(num_classes, 1);

% Set criteria to use for clustering
criteria = {'CalinskiHarabasz','DaviesBouldin','silhouette'};% 'gap'

% Number of clusters
num_clusters = [5 5 5 5 5];
% Loop over each class
for i=1:5
    % Extract descriptors for current class from learning set
    class_descriptors = learning_descriptors(learning_set(:, 101) == i,:);
    class_labels = learning_descriptors(learning_set(:, 101) == i,:);

    % Use unsupervised clustering to determine number of clusters for current class
    eva = evalclusters(class_descriptors,'kmeans','CalinskiHarabasz','KList',1:d_p);
    num_clusters(i) = eva.OptimalK;

    % Use k-means clustering to determine cluster centers for current class
    [~, cluster_centers] = kmeans(class_descriptors, num_clusters(i),"Distance","cosine");
    
    % Store clustering information in cell array
    cluster_info{i} = struct('num_clusters', num_clusters(i), 'cluster_centers', cluster_centers);
end

% Initialize cell array to store mixture of Gaussians models
models = cell(num_classes, 1);

% Loop over each class
for i = classes
    i
    % Extract descriptors for current class from learning set
    class_descriptors = learning_descriptors(learning_set(:, 101) == i,:);

    % Set stats
    options = statset('MaxIter',200);

    % Initialize mixture of Gaussians model for current class
    model = fitgmdist(class_descriptors, num_clusters(i), ...
        'Replicates',1,'Start', 'plus','RegularizationValue', 1e-6,'Options',options);

    % Store mixture of Gaussians model in cell array
    models{i} = model;
end

% Initialize vector to store likelihoods of each class
test_likelihoods = zeros(num_classes, size(test_descriptors, 1));

% Loop over each sample in test data
for i = 1:size(test_descriptors, 1)
    % Loop over each class
    for j = 1:num_classes
        % Compute likelihood of current class
        test_likelihoods(j,i) = pdf(models{j}, test_descriptors(i,:))*priors(j);
    end
end

% Assign class label with highest likelihood to current sample
[~, test_predictions] = max(test_likelihoods,[],1);

% Convert to categorical data
test_labels = categorical(test_labels);
test_predictions = categorical(test_predictions');

test_cm = confusionmat(test_labels,test_predictions);
accuracy=sum(diag(test_cm))/sum(test_cm,"all");
figure;
confusionchart(test_cm, ...
    'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Test');