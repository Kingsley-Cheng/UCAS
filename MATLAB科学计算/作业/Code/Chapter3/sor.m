function y=sor(a,b,w,x0)
D=diag(diag(a));U=-triu(a,1);L=-tril(a,-1);
M=(D-w*L)\((1-w)*D+w*U); f=(D-w*L)\b*w;
y=M*x0+f; n=1;
while norm(y-x0)>=1.0e-6
    x0=y;
    y=M*x0+f;
    n=n+1;
end
