clc;clearvars;
x=zeros(1,30001);
y=zeros(1,30001);
for i=1:30000,
    x(1,i+1)=1+y(1,i)-1.4*x(1,i)^2;
    y(1,i+1)=0.3*x(1,i);
end
plot(x,y,'*')
grid on;
title("Henon plot")