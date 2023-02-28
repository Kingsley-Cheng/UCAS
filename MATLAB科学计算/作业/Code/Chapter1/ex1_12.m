clc,clearvars;
theta=linspace(0,2*pi,10000);
subplot(2,2,1);
rho1=1.0013*theta.^2;
polarplot(theta,rho1)
grid on
title("\rho = 1.0013\theta^2")
subplot(2,2,2)
rho2=cos(7.*theta./2);
polarplot(theta,rho2)
grid on
title("\rho = cos(7\theta/2)")
subplot(2,2,3)
rho3=sin(theta)./theta;
polarplot(theta,rho3)
grid on
title("\rho = sin(\theta)/\theta")
subplot(2,2,4);
rho4=1-cos(7*theta).^3;
polarplot(theta,rho4)
grid on
title("\rho = 1-cos^3(7\theta)")
