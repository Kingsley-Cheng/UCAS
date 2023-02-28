clc,clearvars;
[x,y]=meshgrid(-2:0.01:2);
z = (x.^2+y.^2+x.*y).*exp(-x.^2-y.^2-x.*y);
mesh(x,y,z);
i = find(abs(x.^2+y.^2+x.*y).*exp(-x.^2-y.^2-x.*y)==0);
x(i),y(i),z(i)