clc,clearvars;
syms x y;
f(x,y)=(1-cos(x^2+y^2))/((x^2+y^2)*exp(x^2+y^2));
simplify(taylor(f,[x,y],[1,0],'Order',3))