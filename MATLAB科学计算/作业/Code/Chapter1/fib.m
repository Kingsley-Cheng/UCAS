% function of Fibonacci using recursion
function y=fib(k)
if k == 1 || k==2
    y = 1;
else
    y = fib(k-1)+fib(k-2);
end