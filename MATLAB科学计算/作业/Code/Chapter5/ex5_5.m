clc,clearvars;
syms y t;
y = dsolve('D3y+t*y*D2y+t^2*Dy*y^2=exp(-t*y)','y(0)=2','Dy(0)=0','D2y(0)=0')
t_final=10;
x0 = [2;0;0];
tic;
[t,x] = ode45('eq1',[0,t_final],x0);
time1 = toc
subplot(1,2,1)
plot(t,x(:,1))
title("Ode45算法")
tspan=0:0.01:10;
tic;
[t2,x2]=rk_4('eq1',tspan,x0);
time2 = toc
subplot(1,2,2)
plot(t2,x2(:,1))
title("四阶定步长Runge-Kutta算法,0:0.1:100")