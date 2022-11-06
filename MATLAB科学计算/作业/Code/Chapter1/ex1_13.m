clc,clearvars;
[x,y] = meshgrid(-1:0.1:1);

% 3Dplot z=xy x in [-5,5] , y in [-5,5]
z=x.*y;
subplot(2,2,1)
surf(x,y,z)
title("z=xy---3D Plot")
grid on
xlabel("x")
ylabel("y")
zlabel("z")

% contour plot of z=xy
subplot(2,2,2)
contour(x,y,z,30)
title("z=xy---Contour Plot")

% 3Dplot z=sin(xy) x in [-5,5] , y in [-5,5]
[x,y] = meshgrid(-pi:0.1:pi);
subplot(2,2,3)
z2=sin(x.*y);
surf(x,y,z2)
xlabel("x")
ylabel("y")
zlabel("z")
title("z=sin(xy)---3D Plot")

% contour plot of z=sin(xy)
subplot(2,2,4)
contour(x,y,z2,30)
title("z=sin(xy)---Contour Plot")