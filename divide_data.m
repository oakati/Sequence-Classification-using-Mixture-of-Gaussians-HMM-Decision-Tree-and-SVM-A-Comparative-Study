function [X_new, counter] = divide_data(X,labels_seqIDs,classes,seqIDs)
    [nr,nc] = size(X);
    X_labels = labels_seqIDs(:,1);
    X_seqIDs = labels_seqIDs(:,2);

    current_view=1;
    current_seqID=1;

    X_new=struct;
    counter=ones(length(unique(X_labels)),length(unique(X_seqIDs)));
    for i=1:nr
        X_new.(classes{current_view}).(seqIDs{current_seqID})(counter(current_view,current_seqID),:)=X(i,:);
        counter(current_view,current_seqID)=counter(current_view,current_seqID)+1;

        if((i~=nr) && X_seqIDs(i+1)~=(current_seqID-1))
           current_seqID=current_seqID+1;
        end

        if(current_seqID==(length(unique(X_seqIDs))+1))
           current_view=current_view+1;
           current_seqID=1;
           if(current_view>length(unique(X_labels)))
               break;
           end
        end
    end
    %rows represent class, cols represent seqID
    counter=counter-ones(length(unique(X_labels)),length(unique(X_seqIDs)));
end
