clc;clearvars;
x = -4:0.001:4;
y=exp(-(x+1).^2+pi/2).*sin(5*x+2);
plot(x,y);
hold on;
plot(linspace(-3.5,3.5,100),zeros(100));
i = find(abs(exp(-(x+1).^2+pi/2).*sin(5*x+2))<1e-14);
x(i),y(i)

