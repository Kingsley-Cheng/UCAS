clc,clearvars;
ff=optimset; ff.LargeScale='off'; ff.Display='iter';ff.MaxIter=200;
f = inline(['a(1).*exp(-a(2).*x).*cos(a(3).*x+pi/3)+a(4).*exp(-a(5).*x).*cos(a(6).*x+pi/4)'],'a','x')
x = [1.027   1.319  1.204  0.684  0.984  0.864   0.795  0.753  1.058  0.914  1.011  0.926]';
y = [-8.8797 -5.9644 -7.1057 -8.6905 -9.2509 -9.9224 -9.8899 -9.6364 -8.5883 -9.7277 -9.023 -9.6605]';
[ahat,r,j]=nlinfit(x,y,f,[1;2;3;4;5;6],ff);
ahat,r
y1=f(ahat,x);
plot([y y1])