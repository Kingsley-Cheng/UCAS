format short;
clc,clearvars;
syms x y z;
eqn = [x^2*y^2-z*x*y-4*x^2*y*z^2-x*z^2==0;x*y^3-2*y*z^2-3*x^3*z^2-4*x*z*y^2==0;
    y^2*x-7*x*y^2+3*x*z^2-x^4*z*y==0];
var = [x y z];
[x1,x2,x3] = solve(eqn,var);
x = vpa(x1,4)
y = vpa(x2,4)
z = vpa(x3,4)
subs(eqn,var,[0.3176,-0.1025,0.144])
for i=1:19
subs(eqn,var,[x(i) y(i) z(i)])
end