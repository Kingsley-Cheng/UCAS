clc,clearvars;
syms x;
f = exp(abs(x));
[A,B,F]=fseries(f,x,6,-pi,pi);
A,B,F