clc,clearvars;
syms x;
% 做一步变换，把分母变为 1
f = int((sqrt(x*(x+1)))*(sqrt(x+1)-sqrt(x)),x);
simplify(f)