function [t,y]=shooting(f1,f2,tspan,x0f,varargin)
  t0=tspan(1); tfinal=tspan(2); ga=x0f(1); gb=x0f(2);
  [t,y1]=ode45(f1,tspan,[1;0],varargin); 
  [t,y2]=ode45(f1,tspan,[0;1],varargin);
  [t,yp]=ode45(f2,tspan,[0;0],varargin); 
  m=(gb-ga*y1(end,1)-yp(end,1))/y2(end,1);
  [t,y]=ode45(f2,tspan,[ga;m],varargin);
