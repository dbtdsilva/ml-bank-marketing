% Author: Diogo Silva
% Email Address: dbtds@ua.pt
addpath('support-vector-machine/libsvm-3.21/matlab/')
%data = csvread('data/bank-fixed-full.csv'); 
data = csvread('data/bank-fixed.csv'); 

colorstring = 'rbgry';
%scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))

X = data(:,1:end-1);
Y = not(data(:, end));

% 
% %% TRYING PCA ANALYSIS
% Xfiltered = X;
% %Xfiltered(:,10:20) = [];
% %Xfiltered(:,1:5) = [];
% %Xfiltered(:,11) = [];
% 
% [coeff,score,latent,tsquared] = pca(Xfiltered, 'NumComponents', 2);
% Xcentered = score*coeff';
% 
% % 11, 13, 20 são bosta... clean
% %
% %cmap = [1 0 0; 0 1 0; 0 0 1];
% %colormap(cmap);
% %scatter(score(:,1), score(:,2), 40, Y);
% 
%% APPLYING CROSS VALIDATION
crossvalid = 7;

% duration, pdays, poutcome (12, 14, 16)
% http://www.ijeat.org/attachments/File/v4i4/D3963044415.pdf


global gama 
global sigma
global polyorder
gama = 1;
sigma = 2.47;
polyorder = 3;

minX = min(X);
rangeX = max(X)-minX;
lX = length(X);
Xnorm = (X - repmat(minX, lX, 1)) ./ repmat(rangeX, lX, 1);

%for gama = 1:0.1:5
    stats = zeros(crossvalid, 4);
    for index = 1:crossvalid
        %% COMPUTING SVM
        [ TrainX, TestX ] = splitset(Xnorm, index, crossvalid);
        [ TrainY, TestY ] = splitset(Y, index, crossvalid);
        fprintf('Training.. ');
        %fitcsvm(TrainX, TrainY, 'Standardize', true, 'KernelFunction', 'mrbf')
        svmModel = fitcsvm(TrainX, TrainY, ...
            'CategoricalPredictors', [2,3,4,5,6,7,8,9,10,15], ... % additional set
            'Standardize', true, ...
            'KernelFunction', 'rbf', ...
            'Cost', [0,1.72;1,0], ...            % cost of misclassify
            'KernelScale', sigma, ...
            'Prior', [0.2 0.3], ...%'uniform', ...       % empirical, uniform, ...
            'Solver', 'SMO' ...           % ISDA (w Outl), L1QP, SMO (w/o Outl)
            ... %'PolynomialOrder', 3, ...       % default is 3
            ... %'KernelOffset', 0, ...          % 0 for SMO, 0.1 ISDA
            ... %'KKTTolerance', 0, ...          % defaults: 0 SMO, 1e-3 ISDA
            ... %'ScoreTransform', 'doublelogit', ...   % 'none'-(default),'doublelogit',
            ... %'invlogit','ismax','logit','sign','symmetric','symmetriclogit','symmetricismax'
            ... %'DeltaGradientTolerance', 0, ...% defaults: 0 ISDA, 1e-3 SM
         );

        % poly, SMO
        fprintf('Trained! Predicting.. ');
        out = predict(svmModel, TestX);
        fprintf('Predicted! ');
        %% COMPUTING FMEASURE
        [acc, fscore, recall, precision] = fmeasure(TestY, out);
        stats(index, :) = [fscore, acc, recall, precision];
        fprintf('F-Measure: %.4f, Accuracy: %.4f, Recall: %.4f, Precision: %.4f\n', fscore, acc, recall, precision)
    end
    stats_mean = [sigma mean(stats)]
%end
%stats_std = std(stats)

