clc,clearvars;
x=[9.78, 9.17, 10.06, 10.14, 9.43, 10.60, 10.59, 9.98, 10.16,10.09, 9.91, 10.36]';
[H,p,c,d] = jbtest(x,0.05);
[mu,sigma,delta_mu,delta_s] = normfit(x,0.05)