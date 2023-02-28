clc,clearvars;
syms x y t;
[x,y]=dsolve('D2x(t)+5*Dx(t)+4*x(t)+3*y(t)=exp(-6*t)*sin(4*t)', ...
    '2*Dy(t)+y(t)+4*Dx(t)+6*x(t)=exp(-6*t)*cos(4*t)','x(0)=1','x(pi)=2','y(0)=0');
x = vpa(x,10),y = vpa(y,10)