function [acc, fscore] = fmeasure(Y, Ypred)
% FMEASURE - Computes the f-measure metric.
% This function calculates the f-measure using a vector with the 
% actual classes and the predicted classes.
% Mind that fmeasure considers the class 1 (y == 1) to be the most rare
% class.
    true_positives = sum(and(Y, Ypred));
    true_negatives = sum(and(not(Y), not(Ypred)));
    false_positives = sum(and(not(Y), Ypred));
    false_negatives = sum(and(Y, not(Ypred)));

    acc = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives);
    if true_positives + false_negatives == 0
        recall = 0;
    else
        recall = true_positives / (true_positives + false_negatives);
    end;

    if true_positives + false_positives == 0
        precision = 0;
    else
        precision = true_positives / (true_positives + false_positives);
    end;

    if recall + precision == 0
        fscore = 0;
    else
        fscore = (2 * precision * recall) / (recall + precision);
    end;

    fprintf('Accuracy %.2f\n', acc)
    fprintf('F-measure %f\n', fscore)
    
    