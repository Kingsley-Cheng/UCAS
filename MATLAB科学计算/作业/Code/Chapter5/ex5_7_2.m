clc,clearvars;
syms y1 y2 y3 t s;
% 解析解
[y1,y2,y3] = dsolve('Dy1=-0.1*y1-49.9*y2','Dy2=-50*y2','Dy3=70*y2-120*y3','y1(0)=1', ...
    'y2(0)=2','y3(0)=1')

f1= @(s) 2*exp(-50*s) - exp(-s/10);
f2= @(s) 2*exp(-50*s);
f3= @(s) 2*exp(-50*s) - exp(-120*s);
%普通数值解
x0 = [1;2;1];
t_final =10;
[t,x] = ode45('eq4',[0,t_final],x0);

%刚性数值解
[t2,x2] = ode15s('eq4',[0,t_final],x0);
subplot(2,2,2)
plot3(x(:,1),x(:,2),x(:,3))
title("ode45解")
subplot(2,2,1)
plot3(f1(t),f2(t),f3(t))
title("精确解")
subplot(2,2,3)
plot3(f1(t2),f2(t2),f3(t2))
title("精确解")
subplot(2,2,4)
plot3(x2(:,1),x2(:,2),x2(:,3))
title("ode15s解")