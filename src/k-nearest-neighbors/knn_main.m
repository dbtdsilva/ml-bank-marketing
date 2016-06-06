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
