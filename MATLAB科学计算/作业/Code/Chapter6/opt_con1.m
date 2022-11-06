  function [c,ceq]=opt_con1(x)
   ceq=[];
   c = [-x(1)+x(2)-2;x(1)^2-x(2)+1];
