clc,clearvars;
syms t y;
y=dsolve(['D5y+13*D4y+64*D3y+152*D2y+176*Dy+80*y=','exp(-2*t)*(sin(2*t+pi/3)+cos(3*t))'],'y(0)=1', ...
    'y(1)=3','y(pi)=2','Dy(0)=1','Dy(1)=2');
vpa(y,10)