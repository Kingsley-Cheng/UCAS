clc,clearvars;
A = [10,-1,-2;-1,10,-2;-1,-1,5];
b = [72;83;42];
x0 = [0;0;0];
% jacobi 方法计算方程的解
y1 =jacobi(A,b,x0)
% Gauss-Seidel 方法计算方程的解
y2 = seidel(A,b,x0)