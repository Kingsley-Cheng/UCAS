clc,clearvars;
syms x a;
f = log(x+sqrt(1+x^2));
% 在 0 处展开 4 阶
taylor(f,x,0,'Order',4)
% 在 a 处展开 4 阶
simplify(taylor(f,x,a,'Order',4))