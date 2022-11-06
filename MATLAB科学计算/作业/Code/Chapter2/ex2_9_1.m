clc,clearvars;
syms x t a;
f = int(sin(t)./t,t,0,x);
% 在 0 处展开 4 阶
taylor(f,x,0,'Order',4)
% 在 a 处展开 4 阶
taylor(f,x,a,'Order',4)