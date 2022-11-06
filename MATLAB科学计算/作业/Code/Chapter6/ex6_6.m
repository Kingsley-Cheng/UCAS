clc,clearvars;
[x1,x2]=meshgrid(-3:.1:3); 
z=x1.^3+x2.^2-4*x1+4; 
i=find(-x1.^2+x2-1<0); z(i)=NaN; 
i=find(x1-x2+2<0); z(i)=NaN; 
surf(x1,x2,z);  shading interp;
min(z(:)),view(0,90)


ff=optimset; ff.LargeScale='off'; ff.Display='iter';
ff.TolFun=1e-30; ff.TolX=1e-15; ff.TolCon=1e-20;
x0=[1;1]; xm=[0;0]; xM=[]; A=[]; B=[]; Aeq=[]; Beq=[];
[x,f_opt,c,d]=fmincon('opt_fun1',x0,A,B,Aeq,Beq,xm,xM, 'opt_con1',ff);
x,f_opt


