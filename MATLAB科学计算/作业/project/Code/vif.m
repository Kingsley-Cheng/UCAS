function [diagvalue] = vif(x)
x = corr(x,"type","Pearson");
diagvalue = diag(inv(x));
end