function Ypred = knn_predict(X, Y, Xpred, DistFunct, Kvalue)
% KNN_predict is a function that takes the training data set,the data set that 
% we want to predict, the distance function and K value which is the number of 
% neighboors needed to classify the values. The function will return the 
% predictions.
  Ypred = zeros(rows(Xpred), 1);

  for i = 1:rows(Xpred)
    Ypred(i) = DistFunct(X, Y, Xpred(i, :), Kvalue);
  endfor