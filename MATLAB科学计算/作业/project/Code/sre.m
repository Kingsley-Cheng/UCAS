function [sre,delsre] = sre(x,residual)
n = size(x,1);
m = size(x,2);
b = ones(size(x,1),1);
x = [b,x];
H = x*inv(x'*x)*x';
h = diag(H);
sse=residual'*residual;
sigma=sqrt(sse/(n-m-1));
sre = zeros(size(x,1),1);
for i=1:size(x,1)
    sre(i) = residual(i)/(sigma*sqrt(1-h(i)));
    delsre(i) = sre(i)*sqrt((n-m-2)/(n-m-1-sre(i)^2));
end
end