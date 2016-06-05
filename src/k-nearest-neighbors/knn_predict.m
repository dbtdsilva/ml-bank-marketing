function Ypred = knn_predict(X, Y, Xpred, Kvalue, Alg, Pvalue)
% KNN_predict is a function that takes the training data set,the data set that 
% we want to predict, the distance function and K value which is the number of 
% neighboors needed to classify the values. The function will return the 
% predictions.
  Ypred = zeros(rows(Xpred), 1);

  for i = 1:rows(Xpred)
    if(Alg == 1)
      Ypred(i) = unweighted_chebyshev_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 2)
      Ypred(i) = weighted_chebyshev_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 3)
      Ypred(i) = unweighted_euclidean_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 4)
      Ypred(i) = weighted_euclidean_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 5)
      Ypred(i) = unweighted_manhattan_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 6)
      Ypred(i) = weighted_manhattan_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue);
    elseif(Alg == 7)
      Ypred(i) = unweighted_minkowski_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue, Pvalue);
    elseif(Alg == 8)
      Ypred(i) = weighted_minkowski_distance_knn_predictor(X, Y, Xpred(i, :), Kvalue, Pvalue);     
    endif
  endfor

end