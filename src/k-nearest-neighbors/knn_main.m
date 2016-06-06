% Author: Eduardo Sousa
% Email Address: 	eduardosousa@ua.pt

data = csvread('data/bank-fixed.csv'); 

X = data(:, 1:end-1);
Y = not(data(:, end));

MaxKValue = 100;
KVal = 1:2:MaxKValue;

uwc_stats  = zeros(50, 4);
wc_stats   = zeros(50, 4);
uwe_stats  = zeros(50, 4);
we_stats   = zeros(50, 4);
uwm_stats  = zeros(50, 4);
wm_stats   = zeros(50, 4);
uwmi_stats = zeros(50, 4);
wmi_stats  = zeros(50, 4);

for i = 1:columns(KVal)
  crossvalid = 10;

  % Unweighted Chebyshev 
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 1);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  uwc_stats(i, 1:4) = mean(stats);
  
  % Weighted Chebyshev
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 2);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
   stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  wc_stats(i, 1:4) = mean(stats);
  
  % Unweighted Euclidean
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 3);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  uwe_stats(i, 1:4) = mean(stats);
  
  % Weighted Euclidean
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 4);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  we_stats(i, 1:4) = mean(stats);
  
  % Unweighted Manhattan 
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 5);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
   
  uwm_stats(i, 1:4) = mean(stats);
  
  % Weighted Manhattan
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 6);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  wm_stats(i, 1:4) = mean(stats);
  
  % Unweighted Minkowski
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 7, 10);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  uwmi_stats(i, 1:4) = mean(stats);
  
  % Weighted Minkowski
  stats = zeros(crossvalid, 4);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, KVal(i), 8, 10);
    
    [acc, fscore, recall, precision] = fmeasure(TestY, out);
    stats(index, :) = [acc, fscore, recall, precision];
  endfor
  
  wmi_stats(i, 1:4) = mean(stats);
endfor

% Plotting Accuracy Performance
plot(KVal, uwc_stats(:, 1), 'r-', KVal, wc_stats(:, 1), 'r--', KVal, uwe_stats(:, 1), 'g-', KVal, we_stats(:, 1), 'g--', KVal, uwm_stats(:, 1), 'b-', KVal, wm_stats(:, 1), 'b--', KVal, uwmi_stats(:, 1), 'm-', KVal, wmi_stats(:, 1), 'm--');
xlabel('K');
ylabel('Accuracy');
title('K-Nearest Neighbors Accuracy Performance');
legend('Unweighted Chebyshev Distance', 'Weighted Chebyshev Distance', 'Unweighted Euclidean Distance', 'Weighted Euclidean Distance', 'Unweighted Manhattan Distance', 'Weighted Manhattan Distance', 'Unweighted Minkowski Distance (P=10)', 'Weighted Minkowski Distance (P=10)');
pause;

% Plotting F-Score Performance
plot(KVal, uwc_stats(:, 2), 'r-', KVal, wc_stats(:, 2), 'r--', KVal, uwe_stats(:, 2), 'g-', KVal, we_stats(:, 2), 'g--', KVal, uwm_stats(:, 2), 'b-', KVal, wm_stats(:, 2), 'b--', KVal, uwmi_stats(:, 2), 'm-', KVal, wmi_stats(:, 2), 'm--');
xlabel('K');
ylabel('F-Socre');
title('K-Nearest Neighbors F-Score Performance');
legend('Unweighted Chebyshev Distance', 'Weighted Chebyshev Distance', 'Unweighted Euclidean Distance', 'Weighted Euclidean Distance', 'Unweighted Manhattan Distance', 'Weighted Manhattan Distance', 'Unweighted Minkowski Distance (P=10)', 'Weighted Minkowski Distance (P=10)');
pause;

% Plotting Recall Performance
plot(KVal, uwc_stats(:, 3), 'r-', KVal, wc_stats(:, 3), 'r--', KVal, uwe_stats(:, 3), 'g-', KVal, we_stats(:, 3), 'g--', KVal, uwm_stats(:, 3), 'b-', KVal, wm_stats(:, 3), 'b--', KVal, uwmi_stats(:, 3), 'm-', KVal, wmi_stats(:, 3), 'm--');
xlabel('K');
ylabel('Recall');
title('K-Nearest Neighbors Recall Performance');
legend('Unweighted Chebyshev Distance', 'Weighted Chebyshev Distance', 'Unweighted Euclidean Distance', 'Weighted Euclidean Distance', 'Unweighted Manhattan Distance', 'Weighted Manhattan Distance', 'Unweighted Minkowski Distance (P=10)', 'Weighted Minkowski Distance (P=10)');
pause;

% Plotting Precision Performance
plot(KVal, uwc_stats(:, 4), 'r-', KVal, wc_stats(:, 4), 'r--', KVal, uwe_stats(:, 4), 'g-', KVal, we_stats(:, 4), 'g--', KVal, uwm_stats(:, 4), 'b-', KVal, wm_stats(:, 4), 'b--', KVal, uwmi_stats(:, 4), 'm-', KVal, wmi_stats(:, 4), 'm--');
xlabel('K');
ylabel('Precision');
title('K-Nearest Neighbors Precision Performance');
legend('Unweighted Chebyshev Distance', 'Weighted Chebyshev Distance', 'Unweighted Euclidean Distance', 'Weighted Euclidean Distance', 'Unweighted Manhattan Distance', 'Weighted Manhattan Distance', 'Unweighted Minkowski Distance (P=10)', 'Weighted Minkowski Distance (P=10)');
pause;