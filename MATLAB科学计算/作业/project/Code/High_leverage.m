function [h,h_mean,times] = High_leverage(x,residual)
n = size(x,1);
m = size(x,2);
b = ones(size(x,1),1);
x = [b,x];
H = x*inv(x'*x)*x';
h = diag(H);
h_mean = (m+1)/n;
times = h./h_mean;
end