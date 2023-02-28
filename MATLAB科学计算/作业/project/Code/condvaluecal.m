function [condvalue] = condvaluecal(x)
x = corr(x,type="Pearson");
eigvalue = eig(x);
eigmax = max(eigvalue);
condvalue = sqrt(eigmax./eigvalue);
end