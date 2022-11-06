clc,clearvars;
syms x y;
f = (1-cos(x.^2+y.^2))./((x.^2+y.^2).*exp(x.^2+y.^2));
limit(limit(f,x,0),y,0)