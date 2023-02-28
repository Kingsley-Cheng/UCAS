function [rho_hat,DW] = dwtest(residual)
residual1 = residual(1:end-1,1);
residual2 = residual(2:end,1);
DW = ((residual2-residual1)'*(residual2-residual1))/(residual'*residual);
rho_hat =1-DW/2;
end