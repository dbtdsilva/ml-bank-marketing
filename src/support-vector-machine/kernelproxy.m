function G = kernelproxy(U,V)
%KERNELPROXY Kernel proxy to inject custom parameters
%   Author: Diogo Bastos Tavares da Silva
%   Detail: This function enables the use of lambda expression within
% the SVM library
    global mkernel
    G = mkernel(U,V);
end

