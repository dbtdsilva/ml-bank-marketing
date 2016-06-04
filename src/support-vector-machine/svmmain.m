% Author: Diogo Silva
% Email Address: dbtds@ua.pt

data = csvread('data/bank-fixed.csv'); 
% You must compile libsvm modules before with 'make octave'
%addpath('support-vector-machine/libsvm-3.21/matlab/')

colorstring = 'rbgry';
%scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))

X = data(:,1:end-1);
Y = not(data(:, end));

%% TRYING PCA ANALYSIS
%Xfiltered = X;
%Xfiltered(:,10:20) = [];
%Xfiltered(:,1:5) = [];
%Xfiltered(:,11) = [];

%[coeff,score,latent,tsquared] = pca(Xfiltered, 'NumComponents', 2);
%Xcentered = score*coeff';

% 11, 13, 20 são bosta... clean
%
%cmap = [1 0 0; 0 1 0; 0 0 1];
%colormap(cmap);
%scatter(score(:,1), score(:,2), 40, Y);

%% APPLYING CROSS VALIDATION
crossvalid = 10;

stats = zeros(crossvalid, 2);
for index = 1:crossvalid
    %% COMPUTING SVM
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    svmModel = fitcsvm(TrainX, TrainY, ...
        'Standardize',true, ...
        'KernelFunction','polynomial',...
        'KernelScale','auto', ...
        'OutlierFraction', 0 ...
        ... %'IterationLimit', 1e2 ...
        ... %'BoxConstraint', 2 ...%, ...
     );
    out = predict(svmModel, TestX);
    %% COMPUTING FMEASURE
    %Ypred = zeros(length(Y), 1);
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
    fprintf('F-Measure: %.4f, Accuracy: %.4f\n', fscore, acc)
end
mean(stats)
