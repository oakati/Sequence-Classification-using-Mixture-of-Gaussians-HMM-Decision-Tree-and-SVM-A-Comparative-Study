clc;
clear;
close all;
format shortEng;
warning('on','all');
rng("shuffle");
format compact;

% Load learning and test sets
learning_set = dlmread('instantaneous.txt');
test_set = dlmread('instantaneous_test.txt');

% Extract the descriptors, sequence IDs and labels from the sets
learning_descriptors = learning_set(:, 1:100);
learning_labels = learning_set(:, 101)+1;
learning_seq_ids = learning_set(:, 102)+1;
test_descriptors = test_set(:, 1:100);
test_labels = test_set(:, 101)+1;
test_seq_ids = test_set(:, 102)+1;

% Classes
classes = unique(learning_labels)';

% Set number of classes
num_classes = length(classes);

% PCA
d_p=100;
[learning_descriptors, targets, UW, m, W] = PCA(learning_descriptors', learning_labels', d_p);
learning_descriptors = learning_descriptors';
test_descriptors = (W*test_descriptors')';

% Initialize cell array to store clustering information
cluster_info = cell(1, num_classes);

% Number of clusters
num_clusters = [8 2 3 4 7];

for i = classes
    % Extract descriptors for current class from learning set
    class_descriptors = learning_descriptors(learning_labels == i,:);
    class_labels = learning_descriptors(learning_labels == i,:);

    % Use k-means clustering for current class
    [learning_cluster_indices, cluster_centers{i}, SUMD, D] = kmeans(class_descriptors, ...
        num_clusters(i));
    
    % Store clustering information in cell array
    cluster_info{i} = struct('num_clusters', num_clusters(i), ...
        'cluster_indices',learning_cluster_indices, ...
        'cluster_centers', cluster_centers{i}, ...
        'SUMD', SUMD, ...
        'D', D);
end

% Merge all subcluster centers
cluster_centers = [cluster_info{1}.cluster_centers;...
    cluster_info{2}.cluster_centers;...
    cluster_info{3}.cluster_centers;...
    cluster_info{4}.cluster_centers;...
    cluster_info{5}.cluster_centers];

% Total number of classes
tot_num_clusters = sum(num_clusters);

% Initialize cell array to store HMM models
hmm_models = cell(1, num_classes);

% Map learning descriptors to clusters using nearest neighbor method
learning_cluster_indices = knnsearch(cluster_centers, learning_descriptors);

% Loop over each class
disp("hmmtrain");
for i = classes
    i
    % Extract labels for current class from learning set
    class_seq_ids = learning_seq_ids(learning_labels == i,:);

    % Initialize SEQS
    SEQS = {};

    % Loop over unique sequence IDs in current class
    for j = unique(class_seq_ids)'
        % Extract cluster indices for current sequence
        seqs = learning_cluster_indices(class_seq_ids == j)';

        % Add seqs to SEQS
        SEQS = [SEQS seqs];
    end
    
    SEQ = repmat(learning_cluster_indices,10,1);
    STATES = repmat((learning_seq_ids~=i)+1,10,1);

    % Estimate HMM for current class
    [a_head, b_head] = hmmestimate(SEQ,STATES);
    
    % Train HMM for current class
%     [a_est, b_est] = hmmtrain(SEQS, a_head, b_head, "Tolerance", 1e-7, "Algorithm", "baumwelch");

    % Store HMM model for current class in hmm_models
    hmm_models{i}.a_est = a_head;
    hmm_models{i}.b_est = b_head;
end

% Map test descriptors to clusters using nearest neighbor method
test_states = knnsearch(cluster_centers, test_descriptors);

% Get number of unique sequences in test set
num_test_seqs = length(unique(test_seq_ids));

% Initialize logpseq with zeros
logpseq = zeros(num_test_seqs, num_classes);

% Initialize variable to keep track of current test sequence
current_test_seqs = 1;

% Loop through each class
disp("hmmdecode");
for i = classes
    i
    % Extract labels, seq_ids, states for current class from test set
    class_labels = test_labels(test_labels == i,:);
    class_seq_ids = test_seq_ids(test_labels == i,:);
    class_states = test_states(test_labels == i,:);

    % Loop through each unique sequence in current class
    for j = unique(class_seq_ids)'
        % Extract labels, seq_ids, states for current sequence from class set
        seq_labels = class_labels(class_seq_ids == j,:);
        seq_seq_ids = class_seq_ids(class_seq_ids == j,:);
        seq_states = class_states(class_seq_ids == j,:);

        % Loop through each class
        for k = 1:length(classes)
            % Get HMM model parameters for current class
            a_est = hmm_models{k}.a_est;
            b_est = hmm_models{k}.b_est;

            % Decode sequence using HMM model
            SEQ = seq_states;%(:,k)
            [~, LOGPSEQ] = hmmdecode(SEQ,a_est,b_est);

            % Store log probability of sequence given current class
            logpseq(current_test_seqs,k) = LOGPSEQ;
        end

        % Increment current test sequence counter
        current_test_seqs = current_test_seqs + 1;
    end
end

% Get the maximum log-likelihoods from each row of logpseq
[M, test_seq_predictions] = max(logpseq, [], 2, 'omitnan');

% Get unique combinations of labels and sequence IDs in test set
test_seq_labels = unique([test_labels test_seq_ids], "rows");

% Extract labels from test_seq_labels
test_seq_labels = test_seq_labels(:, 1);

% Convert to categorical data
test_seq_labels = categorical(test_seq_labels);
test_seq_predictions = categorical(test_seq_predictions);

test_cm = confusionmat(test_seq_labels,test_seq_predictions);
accuracy=sum(diag(test_cm))/sum(test_cm,"all");
figure;
confusionchart(test_seq_labels,test_seq_predictions, ...
    "RowSummary","row-normalized", ...
    'ColumnSummary','column-normalized');