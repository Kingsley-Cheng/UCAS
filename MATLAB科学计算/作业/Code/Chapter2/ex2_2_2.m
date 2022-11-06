clc,clearvars;
syms x y;
f=log(x.^2+y.^2)-atan(y./x);
F = -diff(f,x)/diff(f,y);
simplify(F,"Criterion","preferReal")