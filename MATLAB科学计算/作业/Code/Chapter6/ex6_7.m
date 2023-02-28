clc,clearvars;
f=[0 0 0 0 0 1 1]'; Ae=[1,1,1,1,0,0,0;-2,1,-1,0,0,-1,1;0,3,1,0,1,0,1];
Be=[4;1;9]; A=[]; B=[]; xm=[0,0,0,0,0,0,0];
ff=optimset; 
ff.Display='iter';
[x,f_opt,key,c]=linprog(f,A,B,Ae,Be,xm,[],[],ff)
