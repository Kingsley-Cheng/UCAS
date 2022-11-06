clc,clearvars;
syms x y t;
f(x,y) = int(exp(-t.^2),t,0,x.*y);
f1 = (x./y).*diff(diff(f,x),x)-2*diff(diff(f,x),y)+diff(diff(f,y),y);
simplify(f1)