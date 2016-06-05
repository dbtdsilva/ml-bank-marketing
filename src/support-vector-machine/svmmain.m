% Author: Diogo Silva
% Email Address: dbtds@ua.pt
addpath('support-vector-machine/libsvm-3.21/matlab/')
%data = csvread('data/bank-fixed-full.csv'); 
data = csvread('data/bank-fixed.csv'); 

colorstring = 'rbgry';
%scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))

X = data(:,1:end-1);
Y = not(data(:, end)); % inverting -> yes is 1 and no is 0

% 
%% TRYING PCA ANALYSIS
Xfiltered = X;
 
[coeff,score,latent,tsquared] = pca(Xfiltered, 'NumComponents', 2);
Xcentered = score*coeff';

cmap = [1 0 0; 0 1 0; 0 0 1];
colormap(cmap);
scatter(score(:,1), score(:,2), 40, Y);
 

%% Normalization every feature value from 0 to 1
minX = min(X);
rangeX = max(X)-minX;
lX = length(X);
Xnorm = (X - repmat(minX, lX, 1)) ./ repmat(rangeX, lX, 1);

%% Algorithm to train, predict and check the performane
% Run for different scales (it depends from the kernel used):
% linear -> no kernel scaling (it will make no difference)
% polynomial -> gamma
% rbf -> gamma 
stats_mean = [];
%algorithms = {{'linear', 0, }, {'polynomial', 20}, 'rbf'};
for kernelScale = 1:0.1:20
    %% APPLYING CROSS VALIDATION
    stats = zeros(crossvalid, 4);
    crossvalid = 7;
    for index = 1:crossvalid
        %% COMPUTING SVM
        [ TrainX, TestX ] = splitset(Xnorm, index, crossvalid);
        [ TrainY, TestY ] = splitset(Y, index, crossvalid);
        fprintf('Training.. ');
        %fitcsvm(TrainX, TrainY, 'Standardize', true, 'KernelFunction', 'mrbf')
        prior0 = mean(TrainY);
        prior1 = 1 - prior0;
        
        % Measuring the performance time
        tic
        svmModel = fitcsvm(TrainX, TrainY, ...
            'CategoricalPredictors', [2,3,4,5,6,7,8,9,10,15], ... % additional set
            'Standardize', true, ...
            'KernelFunction', 'linear', ...
            ... %'Cost', [0,1.72;1,0], ...            % cost of misclassify
            'KernelScale', kernelScale, ...
            'Prior', [prior0 prior1], ...       % empirical, uniform, ...
            ... %'PolynomialOrder', 3, ...       % default is 3
            'KernelOffset', 0 ...          % 0 for SMO, 0.1 ISDA
        );
        trainTiming = toc
        % poly, SMO
        fprintf('Trained! Predicting.. ');
        tic
        out = predict(svmModel, TestX);
        predictTiming = toc
        fprintf('Predicted! ');
        %% COMPUTING FMEASURE
        [acc, fscore, recall, precision] = fmeasure(TestY, out);
        stats(index, :) = [fscore, acc, recall, precision, trainTiming, predictTiming];
        fprintf('F-Measure: %.4f, Accuracy: %.4f, Recall: %.4f, Precision: %.4f\n', fscore, acc, recall, precision)
    end
    stats_mean(end+1,:) = [kernelScale mean(stats)]
end

