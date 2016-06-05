function Ypred = unweighted_manhattan_distance_knn_predictor(X, Y, Xpred, Kvalue)
% Unweighted Manhattan distance predictor
  
  dist = sum(abs(X-Xpred), 2);
  [sortedValues, sortedIndex] = sort(dist, 'ascend');
  Ypred_0 = sum(Y(sortedIndex(1:Kvalue)) == 0);
  Ypred_1 = sum(Y(sortedIndex(1:Kvalue)) == 1);
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