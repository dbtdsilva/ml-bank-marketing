% Author: Diogo Silva
% Email Address: dbtds@ua.pt

data = csvread('data/car_indexed.data'); 
# You must compile libsvm modules before with 'make octave'
addpath('support-vector-machine/libsvm-3.21/matlab/')

colorClass = ["red"; "orange"; "yellow"; "green"]
scatter(data(:, 1), data(:, 2), colorClass(data(:,7) + 1))