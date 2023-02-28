clc,clearvars;
H = [4,-4;-4,8];f=[-6,-3];
OPT=optimset; OPT.LargeScale='off'; 
A=[1,1;4,1]; B=[3;9]; Aeq=[]; Beq=[]; LB=zeros(2,1);
[x,f_opt]=quadprog(H,f,A,B,Aeq,Beq,LB,[],[],OPT)

[x1,x2]=meshgrid(0:.05:3); 
z=2*x1.^2-4*x1.*x2+4*x2.^2-6*x1-3*x2; 
i=find(x1+x2>3); z(i)=NaN; 
i=find(4*x1+x2>9); z(i)=NaN; 
subplot(1,2,1)
surf(x1,x2,z);  shading interp;
min(z(:)),view(0,90)
subplot(1,2,2)
mesh(x1,x2,z)