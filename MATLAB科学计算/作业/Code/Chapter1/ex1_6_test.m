clc,clearvars;
h=1;
D=2;
x=linspace(-3,3,1000);
y=zeros(1,1000);
for i=1:1000
    y(i) = ex1_6(h,D,x(i));
end
plot(x,y)
ylim([-2,2])
xlim([-3,3])