clc,clearvars;
x=[10.4,10.2,12.0,11.3,10.7,10.6,10.9,10.8,10.2,12.1]';
[mu,sigma,delta_u,delta_s] = normfit(x,0.05);
mu,delta_u