clc;clearvars;
A = [5,7,6,5,1,6,5;2,3,1,0,0,1,4;6,4,2,0,6,4,4;3,9,6,3,6,6,2;
    10,7,6,0,0,7,7;7,2,4,4,0,7,7;4,8,6,7,2,1,7];
B = [3,5,5,0,1,2,3;3,2,5,4,6,2,5;1,2,1,1,3,4,6;3,5,1,5,2,1,2;
    4,1,0,1,2,0,1;-3,-4,-7,3,7,8,12;1,-10,7,-6,8,1,5];
A = sym(A)
% 数值矩阵和符号矩阵相乘
A * B
B = sym(B)