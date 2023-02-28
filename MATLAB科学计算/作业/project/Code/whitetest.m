function [W,pvalue] = whitetest(x,residual)
n=size(x,1);
m=size(x,2);
res2 = residual.^2;
xtest =x;
for i =1:m
    xtest = [xtest,x(:,i).^2];
end
[betat,r2t,adjr2t,Ft,Ftestt,tt,ttestt,residualst] = myregression(xtest,res2);
W=n*r2t;
pvalue = 1-chi2cdf(W,size(xtest,2));
end