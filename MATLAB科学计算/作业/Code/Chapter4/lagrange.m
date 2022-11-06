function y=lagrange(x0,y0,x)
ii=1:length(x0);   y=zeros(size(x));
for i=ii
    ij=find(ii~=i);   y1=1;
    for j=1:length(ij),  y1=y1.*(x-x0(ij(j)));  end
    y=y+y1*y0(i)/prod(x0(i)-x0(ij));
end
