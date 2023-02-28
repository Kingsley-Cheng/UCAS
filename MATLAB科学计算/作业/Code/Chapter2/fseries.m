function [A,B,F]=fseries(f,x,n,a,b)
if nargin==3, a=-pi; b=pi; end
L=(b-a)/2; 
if a+b, f=subs(f,x,x+L+a); end
A=int(f,x,-L,L)/L; B=[]; F=A/2;
for i=1:n
    an=int(f*cos(i*pi*x/L),x,-L,L)/L; 
    bn=int(f*sin(i*pi*x/L),x,-L,L)/L; A=[A, an]; B=[B,bn]; 
    F=F+an*cos(i*pi*x/L)+bn*sin(i*pi*x/L);
end
if a+b, F=subs(F,x,x-L-a); end 
