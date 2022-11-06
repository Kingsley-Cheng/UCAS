clc,clearvars;
syms x;
f = sqrt(x*sin(x)*sqrt(1-exp(x)));
simplify(diff(f),IgnoreAnalyticConstraints=true,Criterion="preferReal")