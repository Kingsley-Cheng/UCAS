clc,clearvars;
syms x y;
[x,y]=solve((x^2+y^2+x*y)*exp(-x^2-y^2-x*y),[x,y])
(x^2+y^2+x*y)*exp(-x^2-y^2-x*y)