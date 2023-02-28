clc,clearvars;

syms x y;
f(x,y) = cos(sqrt(x./y))^(-1);
f1 = diff(diff(f,x),y);
pretty(f1)
f2 = diff(diff(f,y),x);
pretty(f2)
