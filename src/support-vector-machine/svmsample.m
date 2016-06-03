% SVM test with Matlab where two linearly separable sets are generated
% The set 1 is in the interval [-1 0] and the set 2 in [0 1]. In the y axes
% the values are the range [-1 1]

clear; clc

rng(1); % For reproducibility
N= 100;     % Number of points for each data set
c1 = [rand(N,1)-1 rand(N,1)*2-1]; % Set 1
c2 = [rand(N,1) rand(N,1)*2-1]; % Set 2


c3 = [c1;c2];
theclass = ones(2*N,1);
theclass(1:N) = -1;

%Train the SVM Classifier
%svmModel = fitcsvm(c3,theclass);    % Linear kernel
svmModel = fitcsvm(c3,theclass,'KernelFunction','rbf');    % Non-linear kernel

% Predict scores over a grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(c3(:,1)):d:max(c3(:,1)),...
    min(c3(:,2)):d:max(c3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svmModel,xGrid);

% Plot the data and the decision boundary
figure(1)
h(1:2) = gscatter(c3(:,1),c3(:,2),theclass,'rb','.');
hold on
h(3) = plot(c3(svmModel.IsSupportVector,1),c3(svmModel.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
hold off
