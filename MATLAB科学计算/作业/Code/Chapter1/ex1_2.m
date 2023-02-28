tic,
A=rand(500);
B=inv(A);
norm(A*B-eye(500)),
toc,