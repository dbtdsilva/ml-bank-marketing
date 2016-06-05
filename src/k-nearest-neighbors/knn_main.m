% Author: Eduardo Sousa
% Email Address: 	eduardosousa@ua.pt

data = csvread('data/bank-fixed.csv'); 

X = data(:, 1:end-1);
Y = not(data(:, end));

MaxKValue = 100;

uwc_stats  = zeros(50, 6);
wc_stats   = zeros(50, 6);
uwe_stats  = zeros(50, 6);
we_stats   = zeros(50, 6);
uwm_stats  = zeros(50, 6);
wm_stats   = zeros(50, 6);
uwmi_stats = zeros(50, 6);
wmi_stats  = zeros(50, 6);

for i = 1:2:MaxKValue
  crossvalid = 10;

  % Unweighted Chebyshev 
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 1);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  uwc_stats(i, 1:2) = mean(stats);
  uwc_stats(i, 3:4) = std(stats);
  uwc_stats(i, 5:6) = max(stats);
  
  % Weighted Chebyshev
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 2);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  wc_stats(i, 1:2) = mean(stats);
  wc_stats(i, 3:4) = std(stats);
  wc_stats(i, 5:6) = max(stats);
  
  % Unweighted Euclidean
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 3);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  uwe_stats(i, 1:2) = mean(stats);
  uwe_stats(i, 3:4) = std(stats);
  uwe_stats(i, 5:6) = max(stats);
  
  % Weighted Euclidean
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 4);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  we_stats(i, 1:2) = mean(stats);
  we_stats(i, 3:4) = std(stats);
  we_stats(i, 5:6) = max(stats);

  % Unweighted Manhattan 
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 5);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  uwm_stats(i, 1:2) = mean(stats);
  uwm_stats(i, 3:4) = std(stats);
  uwm_stats(i, 5:6) = max(stats);
  
  % Weighted Manhattan
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 6);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  wm_stats(i, 1:2) = mean(stats);
  wm_stats(i, 3:4) = std(stats);
  wm_stats(i, 5:6) = max(stats);
  
  % Unweighted Minkowski
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 7, 3);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  uwmi_stats(i, 1:2) = mean(stats);
  uwmi_stats(i, 3:4) = std(stats);
  uwmi_stats(i, 5:6) = max(stats);
  
  % Weighted Minkowski
  stats = zeros(crossvalid, 2);
  for index = 1:crossvalid
    [ TrainX, TestX ] = splitset(X, index, crossvalid);
    [ TrainY, TestY ] = splitset(Y, index, crossvalid);
    out = knn_predict(TrainX, TrainY, TestX, i, 8, 3);
    
    [acc, fscore] = fmeasure(TestY, out);
    stats(index, :) = [fscore, acc];
  endfor
  
  wmi_stats(i, 1:2) = mean(stats);
  wmi_stats(i, 3:4) = std(stats);
  wmi_stats(i, 5:6) = max(stats);
endfor
