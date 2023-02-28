clc,clearvars;
syms x1 x2 x3 x4;
f(x1,x2,x3,x4)=100*(x2-x1^2)^2+(1-x1)^2+90*(x4-x3^2)+(1-x3^2)^2+...
10.1*((x2-1)^2+(x4-1)^2)+19.8*(x2-1)*(x4-1);
J=jacobian(f,[x1,x2,x3,x4]);
ff.GradObj='on'; 
x=fminunc('fun',[1;1;1;1],ff)
vpa(f(x(1),x(2),x(3),x(4)),6)
