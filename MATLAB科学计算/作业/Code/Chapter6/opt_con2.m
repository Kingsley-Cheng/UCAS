function [c,ceq]=opt_con2(x)
   ceq=[];
   c = [x(1)*x(2)-x(1)-x(2)+1.5;-x(1)*x(2)-10];
