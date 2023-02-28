function [rho,tvalue,pvalue] = spearmantest(x,residual)
n = size(x,1);
[rho,p] = corr(x,abs(residual),"type","Spearman");
tvalue = (sqrt(n-2)*rho)/sqrt(1-rho^2);
pvalue = 2*(1-tcdf(abs(tvalue),n-2));
end