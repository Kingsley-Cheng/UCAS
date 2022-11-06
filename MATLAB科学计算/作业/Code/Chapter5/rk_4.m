function [tout,yout]=rk_4(odefile,tspan,y0) %y0初值列向量
   t0=tspan(1); th=tspan(2);
   if length(tspan)<=3, h=tspan(3);      % tspan=[t0,th,h]
   else, h=tspan(2)-tspan(1); th=tspan(end); 
   end %等间距数组   
   tout=[t0:h:th]';
   yout=[];
   for t=tout'
      k1=h*eval([odefile '(t,y0)']);   % odefile是一个字符串变量，为表示微分方程f( )的文件名。
      k2=h*eval([odefile '(t+h/2,y0+0.5*k1)']);
      k3=h*eval([odefile '(t+h/2,y0+0.5*k2)']);
      k4=h*eval([odefile '(t+h,y0+k3)']);
      y0=y0+(k1+2*k2+2*k3+k4)/6;
      yout=[yout; y0'];
   end