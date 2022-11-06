clc,clearvars;
syms y x;
f = dsolve('D2y-(2-1/x)*Dy+(1-1/x)*y=x^2*exp(-5*x)','y(1)=pi','y(pi)=1','x')

% 数值解
[t,y]=shooting('eq51','eq52',[1,pi],[pi,1]);
plot(t,y)

vpa(norm(y(:,1)-vpa(subs(f,x,t))))
