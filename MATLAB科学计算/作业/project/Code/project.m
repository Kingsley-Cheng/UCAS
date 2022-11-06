format long;
clc,clearvars;
data = readtable("Data/Admission_Predict_2.csv",VariableNamingRule="preserve");
data = data(:,2:9);
name = {'GRE Score', 'TOFEL Score' ,'University Rating','SOP','LOR','CGPA','Research','Chance of Admit'};
temp = data{:,:};
% 判断数据是否有空值
sum(isnan(temp),'all');
% 绘制自变量的直方图
subplot(1,4,1);
histogram(Data =data.("GRE Score"),NumBins=10);
title('GRE.Score');
ylabel('Frequency');
subplot(1,4,2);
histogram(Data =data.("TOEFL Score"),NumBins=10);
title('TOEFL.Score');
ylabel('Frequency');
subplot(1,4,3);
histogram(Data =data.("University Rating"));
title('University.Rating');
ylabel('Frequency');
subplot(1,4,4);
histogram(data.("SOP"));
title('SOP');
ylabel('Frequency');
subplot(1,3,1)
histogram(Data =data.("LOR"));
title('LOR');
ylabel('Frequency');
subplot(1,3,2)
histogram(Data =data.("CGPA"));
title('CGPA');
ylabel('Frequency');
subplot(1,3,3)
histogram(Data =data.("Chance of Admit"));
title('Chance of Admit');
ylabel('Frequency');
% 各个变量的基本统计量
minvalue = min(temp);
maxvalue = max(temp);
averagevalue = mean(temp);
medianvalue = median(temp);
% 绘制各变量的箱形图
subplot(1,4,1);
boxchart(data.("GRE Score"),Notch="on");
title('GRE.Score');
subplot(1,4,2);
boxchart(data.("TOEFL Score"),Notch="on");
title('TOEFL.Score');
subplot(1,4,3);
boxchart(data.("University Rating"),Notch="on");
title('University.Rating');
subplot(1,4,4);
boxchart(data.("SOP"),Notch="on");
title('SOP');
subplot(1,3,1)
boxchart(data.("LOR"),Notch="on");
title('LOR');
subplot(1,3,2)
boxchart(data.("CGPA"),Notch="on");
title('CGPA');
subplot(1,3,3)
boxchart(data.("Chance of Admit"),Notch="on");
title('Chance of Admit');
% 删除异常值
data([93,348,377],:)=[];
close all;
% 绘制各个变量间的相关热力图
subplot(1,1,1);
corr_matrix = corr(temp);
heatmap(corr_matrix,"ColorbarVisible","on","ColorData",corr_matrix,"Colormap",parula,XDisplayLabels=name,YDisplayLabels=name);

% 建立线性回归模型
x=table2array(data(:,1:7));
y=table2array(data(:,8));
n=length(y);
b = ones(n,1);
x_zeros = [b,x];
[beta,r2,adjr2,F,Ftest,t,ttest,residuals] = myregression(x,y);
% 逐步回归
stepwiselm(x,y,Criterion="aic",Upper="linear");

% Lasso 回归
[B,FitInfo] = lasso(x,y,'CV',10,'Alpha',1); 
lassoPlot(B,FitInfo, 'PlotType' , 'CV' );
legend( 'show' ) % 显示图例
% 筛选稀疏变量
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);%回归系数
coef0 = FitInfo.Intercept(idxLambda1SE);%常系数
lanmda=FitInfo.LambdaMinMSE;

% elastic net
[B,FitInfo] = lasso(x,y,'CV',10,'Alpha',0.75); 
lassoPlot(B,FitInfo, 'PlotType' , 'CV' );
legend( 'show' ) % 显示图例
% 筛选稀疏变量
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);%回归系数
coef0 = FitInfo.Intercept(idxLambda1SE);%常系数
lanmda=FitInfo.LambdaMinMSE;
close all;
% 异方差检验
% 画Y与残差的散点图
x = x(:,[1,2,3,5,6,7]);
[beta,r2,adjr2,F,Ftest,t,ttest,residuals] = myregression(x,y);
scatter(y,residuals);
xlabel("Y");
ylabel("Residuals");
close all;
% spearman检验
[rho,tvalue,pvalue]=spearmantest(x(:,1),residuals);
% white检验
[W,pvalue] = whitetest(x,residuals);
% Box-Cox 变换
% 求对数似然下最大的lambda值
lambda = -5:0.1:5;
SSE_lam=arrayfun(@(t) (bocc(x_zeros,y,t)),lambda);
[value,index] = min(SSE_lam);
lambda_min=lambda(index);
SSE_min = SSE_lam(index);
plot(lambda,SSE_lam,'g',lambda_min,SSE_min,'r+');
xlabel('\lambda');ylabel('残差平方和');
legend('不同\lambda下SSE值','\lambda的最优解');
close all;
% 做 BOX-COX 变换
y = boxcox(lambda_min,y);
% 重新拟合
[beta,r2,adjr2,F,Ftest,t,ttest,residuals] = myregression(x,y);
% 画Y与残差的散点图
scatter(y,residuals);
xlabel("Y");
ylabel("Residuals");
close all;
% spearman检验
[rho,tvalue,pvalue]=spearmantest(x(:,1),residuals);
% white检验
[W,pvalue] = whitetest(x,residuals);

% 自相关检验
% 残差时序图
scatter(residuals(1:end-1),residuals(2:end));
xlabel("Residual_{t-1}");
ylabel("Residual_t");
close all;
%DW检验
[rho_hat,DW]=dwtest(residuals);
%wls变换
newx = x(2:end,:)-rho_hat.*x(1:end-1,:);
newy = y(2:end,:)-rho_hat.*y(1:end-1,:);
[beta,r2,adjr2,F,Ftest,t,ttest,residuals] = myregression(newx,newy);
scatter(residuals(1:end-1),residuals(2:end));
xlabel("Residual_{t-1}");
ylabel("Residual_t");
close all;
%DW检验
[rho_hat,DW]=dwtest(residuals);

% 共线性检验
diagvalue = vif(newx);
condvalue = condvaluecal(newx);

% 删除学生残差
[sre,delsre] =sre(newx,residuals);
scatter(1:size(delsre,2),abs(delsre));
hold on;
plot(1:size(delsre,2),3*ones(size(delsre,2),2),'r--');
ylabel("SRE")
close all;
% cook distance
D = cook(newx,residuals);
n= size(D,1);
scatter(1:n,D);
ylabel("Cook Distance")
close all;
% 杠杆值点
[h,h_mean,times] = High_leverage(newx,residuals);
scatter(1:n,times);
ylabel("High leverage times")
hold on;
plot(1:n,2*ones(n,1),'r--',1:n,3*ones(n,1),'r-');
% close all
