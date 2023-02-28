clc,clearvars;
syms k n;
% 前 n 项的和
symsum((1/2).^k+(1/3).^k,k,1,n)
% 无穷项的和
symsum((1/2)^k+(1/3)^k,k,1,inf)