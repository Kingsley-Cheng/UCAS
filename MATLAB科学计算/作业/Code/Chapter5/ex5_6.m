clc,clearvars;
%解析解
syms x y t;
[x,y] = dsolve('D2x(t)=-2*x(t)-3*Dx(t)+exp(-5*t)','D2y(t)=2*x(t)-3*y(t)-4*Dx(t)-4*Dy(t)-sin(t)' ...
    ,'x(0)=1','Dx(0)=2','y(0)=3','Dy(0)=4')
% 数值解：
x0 = [1;2;3;4];
t_final=10;
[t,x] = ode45('eq2',[0,t_final],x0);
subplot(1,2,1)
plot(t,x(:,3))
xlabel("x")
ylabel("y")
title("数值解")
f1 = @(t)  exp(-2*t).*(exp(-3*t)/3 - 10/3) - 2*exp(-t).*(exp(-4*t)/8 - 17/8);
f2 = @(t) (8*exp(-t) - 6.*t.*exp(-t)).*(exp(-4*t)/8 - 17/8) - 10*exp(-2*t).*(exp(-3*t)/3 - 10/3) - 2*exp(-3*t).*((exp(3*t).*(cos(t) - 3*sin(t)))/40 - (7*exp(-2*t))/8 + 71/10) + 6*exp(-t).*((exp(t).*((13*exp(-5*t))/8 + cos(t)/2 - sin(t)/2 + (3.*t.*exp(-5*t))/2))/12 + 7/96);
x1 = f1(t);
y1 = f2(t);
subplot(1,2,2)
plot(t,y1)
xlabel("x")
ylabel("y")
title("解析解")

