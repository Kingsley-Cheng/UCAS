format short;
clc,clearvars;
A = [2,-9,3,-2,-1;10,-1,10,5,0;8,-2,-4,-6,3;-5,-6,-6,-8,-4];
b = [-1,-4,0;-3,-8,-4;0,3,3;9,-5,3];
rank(A),rank([A b])
% 有无穷多解

% 数值解
x = A\b
x2 = null(A)
x3 = x+x2
% 验证数值解的正确性
norm(A*x3-b)
%解析解
%特解
x4 = pinv(sym(A))*b
%基础解系
x5 = null(sym(A))
x6 = x4+x5
norm(A*x6-b)