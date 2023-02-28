clc,clearvars;
syms x ;
f(x) = x.^5+3*x.^4+4*x.^3+2*x.^2+3*x+6;
f(x)
syms s;
x = (s-1)./(s+1);
f(x)
simplify(f(x))