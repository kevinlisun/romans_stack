function [precision recall fscore] = evaluate_seg_result(pred, gt, C)

precision = zeros(1,C+1);
recall = zeros(1,C+1);
fscore = zeros(1,C+1);

pred = double(pred);
gt = double(gt);

TP_ALL = 0;
FP_ALL = 0;
FN_ALL = 0;
TN_ALL = 0;

for i =2:C
    label = i-1;
    
    if sum(gt==label)==0
        continue;
    end
    
    predi = pred(:);
    predi(predi~=label) = NaN;
    gti = gt(:);
    gti(gti~=label) = NaN;
    
    TP = sum(predi==label & gti == label);
    FP = sum(predi==label & gti ~= label);
    FN = sum(predi~=label & gti == label);
    TN = sum(predi~=label & gti ~= label);
    
    TP_ALL = TP_ALL + TP;
    FP_ALL = FP_ALL + FP;
    FN_ALL = FN_ALL + FN;
    TN_ALL = TN_ALL + TN;
    
    precision(i) = TP / (TP+FP);
    
    recall(i) = TP / (TP+FN);
    
    fscore(i) = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i));
    
end


precision(C+1) = TP_ALL / (TP_ALL+FP_ALL);

recall(C+1) = TP_ALL / (TP_ALL+FN_ALL);

fscore(C+1) = 2 * (precision(C+1)*recall(C+1)) / (precision(C+1)+recall(C+1));

    
    