clc,clearvars;

[x,y]=meshgrid(-1:0.01:1);
z = sin(x.*y);
i=find(x.^2+y.^2<=0.5^2); z(i)=NaN; 
mesh(x,y,z)