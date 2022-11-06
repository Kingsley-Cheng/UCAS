clc,clearvars;
syms x y;
eqn = [x^2-y-1==0,(x-2)^2+(y-0.5)^2-1==0];
[x1,x2] = solve(eqn,[x y],"Real",true);
x1 = vpa(x1)
x2 = vpa(x2)

subs(eqn, [x y], [x1(1) x2(1)])
subs(eqn, [x y], [x1(2) x2(2)])