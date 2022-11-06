% Code for mat_add test

clc;clearvars;
A = [1,2;3,4];
B = [3,4;5,6];
C = [5,6;7,8];
S1 = mat_add(A,B)
S2 = mat_add(A,B,C)