clc,clearvars;
% 生成等距的 50 个点
t = linspace(0,3,50);
y = sin(10*t.^2+3);
ti = linspace(0,3,1000);
% 使用一维数据进行差值
yi = interp1(t,y,ti,"spline");
ys = sin(10*ti.^2+3);
%画出精确值
plot(ti,yi,'b--',ti,ys,'r-');
legend("interp1","sin(10t^2+3)")
