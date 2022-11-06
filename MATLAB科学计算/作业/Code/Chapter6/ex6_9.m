clc,clearvars
ff=optimset; ff.LargeScale='off'; ff.Display='iter';
ff.TolFun=1e-30; ff.TolX=1e-15; ff.TolCon=1e-20;
x0=[0;0]; xm=[-10;-10]; xM=[10,10]; A=[1,1]; B=[0]; Aeq=[]; Beq=[];
[x,f_opt,c,d]=fmincon('opt_fun2',x0,A,B,Aeq,Beq,xm,xM, 'opt_con2',ff);
x,f_opt