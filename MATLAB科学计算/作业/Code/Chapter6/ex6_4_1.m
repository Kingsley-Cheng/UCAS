clc,clearvars;
syms x;
x = solve(exp(-(x+1)^2+pi/2)*sin(5*x+2),x)
exp(-(x+1)^2+pi/2)*sin(5*x+2)