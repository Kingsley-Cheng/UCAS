clc,clearvars;
syms x a;
f = exp(-5*x)*sin(3*x+pi/3);
% 在 0 处展开 4 阶
taylor(f,x,0,'Order',4)
% 在 a 处展开 4 阶
simplify(taylor(f,x,a,'Order',4))