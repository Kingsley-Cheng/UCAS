function [D] = cook(x,residual)
n = size(x,1);
m = size(x,2);
b = ones(size(x,1),1);
x = [b,x];
H = x*inv(x'*x)*x';
h = diag(H);
sse=residual'*residual;
sigma=sqrt(sse/(n-m-1));
D = zeros(n,1);
for i=1:n
    D(i) = ((residual(i)^2)/((m+1)*sigma^2))*(h(i)/((1-h(i))^2));
end
end