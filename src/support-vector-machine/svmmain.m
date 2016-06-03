% Author: Diogo Silva
% Email Address: dbtds@ua.pt

data = csvread('data/bank-fixed.csv'); 
% You must compile libsvm modules before with 'make octave'
%addpath('support-vector-machine/libsvm-3.21/matlab/')

colorstring = 'rbgry';
%scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))

X = data(:,1:end-1);

%% TRYING PCA ANALYSIS
%Xfiltered = X;
%Xfiltered(:,10:20) = [];
%Xfiltered(:,1:5) = [];
%Xfiltered(:,11) = [];

%[coeff,score,latent,tsquared] = pca(Xfiltered, 'NumComponents', 2);
%Xcentered = score*coeff';

% 11, 13, 20 são bosta... clean
Y = data(:, end);
cmap = [1 0 0; 0 1 0; 0 0 1];
colormap(cmap);
scatter(score(:,1), score(:,2), 40, Y);

%% COMPUTING SV
