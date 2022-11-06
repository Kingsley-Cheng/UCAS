clc,clearvars;
x=[-2,-1.7,-1.4,-1.1,-0.8,-0.5,-0.2,0.1,0.4,0.7,1,1.3,...
1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.7,4,4.3,4.6,4.9];

y=[0.10289,0.11741,0.13158,0.14483,0.15656,0.16622,0.17332,...
0.1775,0.17853,0.17635,0.17109,0.16302,0.15255,0.1402,...
0.12655,0.11219,0.09768,0.08353,0.07019,0.05786,0.04687,...
0.03729,0.02914,0.02236];

n=15;
xi = linspace(-2,4.9,n);
% 真实数据点
subplot(1,3,1)
scatter(x,y);
title("真实数据点")
% 一维数据差值
subplot(1,3,2)
yi1 = interp1(x,y,xi);
plot(xi, yi1,'b');
title("一维数据差值");
hold on;
scatter(x,y);
% Lagrange插值
yi2 = lagrange(x,y,xi);
subplot(1,3,3)
plot(xi,yi2,'b');
title("Lagrange 插值");
hold on;
scatter(x,y);