clc,clearvars;
syms x n;
f = (pi - abs(x))*sin(x);
[A,B,F] = fseries(f,x,6,-pi,pi);
A,
B,
F,