clc;clearvars;

%方法一：采用循环计算
s1=0;
for i=0:63
    s1=s1+2^i;
end
s1

%方法二：不采用循环
2^64-1

%方法三：使用符号计算
syms x;
f(x)=2.^x-1;
subs(f(x),64)