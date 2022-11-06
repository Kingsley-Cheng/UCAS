clc,clearvars;
p = randn(1,30000)*1.4+0.5;
xx=-4:.5:6; 
yy=hist(p,xx); 
yy=yy/(30000*0.5);
bar(xx,yy);
y = normpdf(xx,0.5,1.4);
line(xx,y);
mu = mean(p)
sigma = std(p)
