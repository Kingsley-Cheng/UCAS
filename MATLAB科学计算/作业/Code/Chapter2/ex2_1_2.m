clc,clearvars;
syms x y;
f = (x.^2.*y+x.*y.^3)/(x+y).^3;
limit((limit(f,x,-1)),y,2)