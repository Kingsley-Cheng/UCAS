clc,clearvars;
syms x;
f = (x+2).^(x+2).*(x+3).^(x+3)./(x+5).^(2.*x+5);
limit(f,x,inf)