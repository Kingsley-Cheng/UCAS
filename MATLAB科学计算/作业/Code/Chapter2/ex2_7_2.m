clc,clearvars;
syms x;
f = (1+x.^2)./(1+x.^4);
I = simplify(int(f,x,0,1));
I