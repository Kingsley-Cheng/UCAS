function [y]=ex1_6(h,D,x)
if(x>D)
    y=h;
elseif(abs(x)<=D)
    y=h./D.*x;
else
    y=-h;
end