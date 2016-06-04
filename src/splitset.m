function [Train, Test] = splitset(set, index, N)
%CROSSVALIDATE Splits a set into a training set and test set.
%   set - Original data set
%   index - Index to be fetched from the split.
%   N - Number of splits to be applied to the data set.
    if index == 1
        Train = set;
        Test = set;
        return
    end
    div = length(set) / N;
    Test = set(round(div * (index-1))+1:round(div * index), :);
    set(round(div * (index-1))+1:round(div * index), :) = [];
    Train = set;
end

