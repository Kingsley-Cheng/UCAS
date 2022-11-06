clc,clearvars;
syms x;
f = (cos(x))./sqrt(x);
I = simplify(int(f,x,0,inf));
I