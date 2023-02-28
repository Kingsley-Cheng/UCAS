clc;clearvars;

% Equal Interval
subplot(1,2,1);
t1 = linspace(-1,1,1000);
y1 = sin(1./t1);
plot(t1,y1)
grid on;
title('Sin(1/t)(Equal Interval)');

% Ineqal Interval
subplot(1,2,2);
% Use rand() product Inequal Interval 
t2 = sort(rand(1,1000).*2-1);
y2 = sin(1./t2);
plot(t2,y2)
grid on;
title('Sin(1/t)(Inqual Interval)');