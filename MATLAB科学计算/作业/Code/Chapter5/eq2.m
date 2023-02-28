function y1 = eq2(t,x)
y1 = [x(2);-2*x(1)-3*x(2)+exp(-5*t);x(4);2*x(1)-3*x(3)-4*x(2)-4*x(4)-sin(t)];
end