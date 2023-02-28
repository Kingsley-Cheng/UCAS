function SSE=bocc(X,Y,lambda)
H=X*inv(X'*X)*X';
n=length(Y);
switch lambda
    case 0
       z=log(Y)*prod(Y)^(1/n);
    otherwise
       z=(Y.^lambda-1)/lambda/(prod(Y)^((lambda-1)/n));
end
SSE=z'*(eye(n)-H)*z;
end