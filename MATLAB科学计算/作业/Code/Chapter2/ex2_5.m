clc,clearvars;
syms x y z;
F = [3*x+exp(y).*z;x.^3+y.^2.*sin(z)];
f = jacobian(F,[x,y,z]);
simplify(f)