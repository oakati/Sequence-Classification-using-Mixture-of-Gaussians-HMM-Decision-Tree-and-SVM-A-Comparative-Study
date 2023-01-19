clear
clc
close all
rng ("shuffle")
 
%%Load dataset
train_set = importdata("instantaneous.txt"); 

%Load test dataset
test_set = importdata("instantaneous_test.txt"); 

%Extract label information
labels_seqIDs = train_set(:, size(train_set, 2)-1:size(train_set, 2)); %sequences are not used, just for our previous defined func needed them
labels = labels_seqIDs(:,1); 
labels_seqIDs_TEST = test_set(:, size(test_set, 2)-1:size(test_set, 2));
labels_TEST = labels_seqIDs_TEST(:,1); 

train_set=train_set(:, 1:100);
test_set=test_set(:, 1:100);

%TREE CONSTRUCTION
%USING FITCTREE
tree=fitctree(train_set,labels,"CrossVal","on","MergeLeaves","on","CategoricalPredictors","all");
view(tree.Trained{1},"Mode","graph");

classes = {'Chair', 'ChairSet', 'Sofa', 'Human', 'Desk'}; 
seqIDs = {'seqID0', 'seqID1', 'seqID2', 'seqID3', 'seqID4'}; 
[X_set, counter_train] = divide_data(train_set, labels_seqIDs, classes, seqIDs); 

%Extract class train informations, for easiness of svm train
Class1 = [X_set.Chair.seqID0; X_set.Chair.seqID1; X_set.Chair.seqID2; X_set.Chair.seqID3; X_set.Chair.seqID4]; 
labels1=zeros(length(Class1),1);
Class2 = [X_set.ChairSet.seqID0; X_set.ChairSet.seqID1; X_set.ChairSet.seqID2; X_set.ChairSet.seqID3; X_set.ChairSet.seqID4]; 
labels2=zeros(length(Class2),1);
Class3 = [X_set.Sofa.seqID0; X_set.Sofa.seqID1; X_set.Sofa.seqID2; X_set.Sofa.seqID3; X_set.Sofa.seqID4]; 
labels3=zeros(length(Class3),1);
Class4 = [X_set.Human.seqID0; X_set.Human.seqID1; X_set.Human.seqID2; X_set.Human.seqID3; X_set.Human.seqID4]; 
labels4=zeros(length(Class4),1);
Class5 = [X_set.Desk.seqID0; X_set.Desk.seqID1; X_set.Desk.seqID2; X_set.Desk.seqID3; X_set.Desk.seqID4]; 
labels5=zeros(length(Class5),1);

SVMModels=cell(4,1);
SVMModels{1}=fitcsvm([Class1;Class2;Class3;Class4;Class5],[labels1+1;labels2;labels3;labels4;labels5],'ClassNames',[false true],'Standardize',true,'KernelFunction','linear');
SVMModels{2}=fitcsvm([Class2;Class3;Class4;Class5],[labels2;labels3;labels4+1;labels5],'ClassNames',[false true],'Standardize',true,'KernelFunction','linear');
SVMModels{3}=fitcsvm([Class2;Class3;Class5],[labels2;labels3;labels5+1],'ClassNames',[false true],'Standardize',true,'KernelFunction','linear');
SVMModels{4}=fitcsvm([Class2;Class3],[labels2+1;labels3],'ClassNames',[false true],'Standardize',true,'KernelFunction','linear');

[~,Scores1]=predict(SVMModels{1},test_set);
[~,Scores2]=predict(SVMModels{2},test_set); 
[~,Scores3]=predict(SVMModels{3},test_set); 
[~,Scores4]=predict(SVMModels{4},test_set); 

predictions=zeros(length(test_set),1);
for i=1:length(test_set)
    if Scores1(i,2)>0
        predictions(i,1)=0;
    else
        if Scores2(i,2)>0
            predictions(i,1)=3;
        else
            if Scores3(i,2)>0
                predictions(i,1)=4;
            else
                if Scores4(i,2)>0
                    predictions(i,1)=1;
                else
                    predictions(i,1)=2;
                end
            end
        end
    end
end
test_cm = confusionmat(labels_TEST,predictions);
accuracy=sum(diag(test_cm))/sum(test_cm,"all");
figure;
conf_matrix_TEST = confusionchart(labels_TEST,predictions);
conf_matrix_TEST.RowSummary = 'row-normalized';
conf_matrix_TEST.ColumnSummary = 'column-normalized';
title(strcat("Confusion matrix for Test set using Supervised decision tree and SVM"));
[prec_TEST,overall_prec_TEST,recall_TEST,overall_recall_TEST]=ROCfromConf(conf_matrix_TEST.NormalizedValues);
prec_TEST
recall_TEST
figure
hold on
plot(recall_TEST,prec_TEST,"*","LineWidth",3,"Color","red");
x_optimal=linspace(0,1,100);
y_optimal=linspace(0,1,100);
line(x_optimal,y_optimal,"LineWidth",1,"Color","blue");
title("ROC Curve for Test set using Supervised decision tree and SVM");
ylabel("Precision for test set");
xlabel("Recall for test set")
legend("Model","Optimal","Location","northwest")
hold off