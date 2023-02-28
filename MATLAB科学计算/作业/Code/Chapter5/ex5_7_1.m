clc,clearvars;
% 解析解
syms y1 y2 t;
[y1,y2] = dsolve('Dy1(t)=9*y1+24*y2+5*cos(t)-1/3*sin(t)','Dy2(t)=-24*y1(t)-51*y2(t)-9*cos(t)+1/3*sint(t)', ...
    'y1(0)=1/3','y2(0)=2/3')

% 常规微分方程解法
t_final =10;
x0 = [1/3;2/3];
[t,x] = ode45('eq3',[0,t_final],x0);
% 刚性微分方程解法
[t2,x2] = ode15s('eq3',[0,t_final],x0);
subplot(2,2,1)
plot(t,x)
xlabel("t")
ylabel("y1,y2")
legend('y1','y2')
title("ode45")
subplot(2,2,2)
plot(x(:,1),x(:,2))
title('ode45')
xlabel('y1')
ylabel('y2')
subplot(2,2,3)
plot(t2,x2)
xlabel("t")
ylabel("y1,y2")
legend('y1','y2')
title("ode15s")
subplot(2,2,4)
plot(x2(:,1),x2(:,2))
title('ode15s')
xlabel('y1')
ylabel('y2')