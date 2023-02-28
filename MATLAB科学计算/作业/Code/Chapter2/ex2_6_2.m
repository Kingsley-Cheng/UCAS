clc,clearvars;
syms x a b;
f = int(x*exp(a*x)*cos(b*x),x);
simplify(f)