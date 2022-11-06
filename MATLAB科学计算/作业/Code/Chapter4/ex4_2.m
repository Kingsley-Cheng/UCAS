clc,clearvars;
% 定义函数
f= @(x,y) (1./(3*x.^3+y)).*exp(-x.^2-y.^4).*sin(x.*y.^2+x.^2.*y);
% 生成等距网格点
[x,y]=meshgrid(0.2:0.2:2);
z = f(x,y);
% 插值
[xe,ye] = meshgrid(0.2:0.02:2);
ze = interp2(x,y,z,xe,ye,"spline");
% 真实值
z2 = f(xe,ye);
% 画出等距插值得到的曲面
subplot(2,2,1)
mesh(xe,ye,ze);
title("等距网格插值")
% 画出误差
subplot(2,2,3)
mesh(xe,ye,abs(ze-z2))
title("与真实值的误差")
% 生成随机网格点
x3 = 4*(rand(1,91))-2;
y3 = 4*(rand(1,91))-2;
z3 = f(x3,y3);
% 插值
ze2 = griddata(x3,y3,z3,xe,ye,"v4");
% 画出随机差值得到的曲面
subplot(2,2,2);
mesh(xe,ye,ze2);
title("随机网格差值");
% 画出误差
subplot(2,2,4)
mesh(xe,ye,abs(ze2-z2))
title("与真实值的误差")


