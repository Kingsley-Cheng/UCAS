function y1 = eq1(t,x)
y1 = [x(2);x(3);exp(-t*x(1))-t*x(1)*x(3)-t^2*x(2)*x(1)^2];
end