clc,clearvars;
syms y x;
% 通解
y = dsolve('D2y-(2-1/x)*Dy+(1-1/x)*y=x^2*exp(-5*x)','x')
% 满足边界条件的解
y1 = dsolve('D2y-(2-1/x)*Dy+(1-1/x)*y=x^2*exp(-5*x)','y(1)=pi','y(pi)=1','x')
