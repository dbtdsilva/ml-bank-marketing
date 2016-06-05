function Ypred = weighted_euclidean_distance_knn_predictor(X, Y, Xpred, Kvalue)
% Weighted Euclidean distance predictor
  
  dist = sqrt(sum((X-Xpred) .^ 2, 2));
  [sortedValues, sortedIndex] = sort(dist, 'ascend');
  Ypred_0 = sum(1 ./ (dist(find(Y(sortedIndex(1:Kvalue)) == 0)) + 0.00000001));
  Ypred_1 = sum(1 ./ (dist(find(Y(sortedIndex(1:Kvalue)) == 1)) + 0.00000001));
  Ycount_0 = sum(Y == 0);
  Ycount_1 = sum(Y == 1);
  
  if (Ypred_0 > Ypred_1)
    Ypred = 0;
  else if (Ypred_1 > Ypred_0)
    Ypred = 1;
  else
    if (Ycount_0 >= Ycount_1)
      Ypred = 0;
    else
      Ypred = 1;
    endif
  endif

end