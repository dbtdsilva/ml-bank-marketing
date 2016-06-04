% Author: Diogo Silva
% Email Address: dbtds@ua.pt

data = csvread('data/bank-fixed-full.csv'); 
% You must compile libsvm modules before with 'make octave'
%addpath('support-vector-machine/libsvm-3.21/matlab/')

colorstring = 'rbgry';
%scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))

X = data(:,1:end-1);
Y = not(data(:, end));

%% TRYING PCA ANALYSIS
Xfiltered = X;
%Xfiltered(:,10:20) = [];
%Xfiltered(:,1:5) = [];
%Xfiltered(:,11) = [];

[coeff,score,latent,tsquared] = pca(Xfiltered, 'NumComponents', 2);
Xcentered = score*coeff';

% 11, 13, 20 são bosta... clean
%
%cmap = [1 0 0; 0 1 0; 0 0 1];
%colormap(cmap);
%scatter(score(:,1), score(:,2), 40, Y);

%% APPLYING CROSS VALIDATION
crossvalid = 7;

X(:, 17:20) = [];
X(:, 12:15) = [];
%X(:, 10) = [];
%X(:, 5:8) = [];
X(:, 1:10) = []; % 4--,2--
stats = zeros(crossvalid, 2);
for index = 1:crossvalid
    %% COMPUTING SVM
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    fprintf('Training.. ');
    svmModel = fitcsvm(TrainX, TrainY, ...
        ... %'Alpha', 0.5 * ones(size(X,1),1), ...   % 0.5 is default for one-class
        ... %'Weights', ones(size(TrainX,1),1), ...       % default is ones (default)
        'CategoricalPredictors', [], ... % 2,3,4,5,6,7,8,9,10,15
        'Standardize', true, ...
        'KernelFunction','rbf',...      % rbf, linear, polynomial
        'Cost', [0,2;1,0], ...
        ... %'PolynomialOrder', 3, ...       % default is 3
        'KernelScale','auto', ...
        ... %'OutlierFraction', 0, ...
        ... %'BoxConstraint', 2, ...         % default is 1
        ... %'ClipAlphas', true, ...         % default is true
        ... %'CacheSize','maximal', ...      % default is 1000
        ... %'GapTolerance', 0.3, ...          % default is 0
        'Prior', 'uniform', ...       % empirical, uniform, ...
        ... %'KernelOffset', 0, ...          % 0 for SMO, 0.1 ISDA
        ... %'KKTTolerance', 0, ...          % defaults: 0 SMO, 1e-3 ISDA
        ... %'ScoreTransform', 'none', ...   % 'none'-(default),'doublelogit',
        ... %'invlogit','ismax','logit','sign','symmetric','symmetriclogit','symmetricismax'
        ... %'DeltaGradientTolerance', 0, ...% defaults: 0 ISDA, 1e-3 SM
        'Solver', 'SMO' ...           % ISDA (w Outl), L1QP, SMO (w/o Outl)
     );
 
    % poly, SMO
    fprintf('Trained! Predicting.. ');
    out = predict(svmModel, TestX);
    fprintf('Predicted! ');
    %% COMPUTING FMEASURE
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
    fprintf('F-Measure: %.4f, Accuracy: %.4f\n', fscore, acc)
end
stats_mean = mean(stats)
stats_std = std(stats)
