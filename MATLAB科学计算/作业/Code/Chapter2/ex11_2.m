clc,clearvars;
syms n k;
f = symsum(1/(n+(k*pi)/n),k,1,n)
limit(f,n,inf)