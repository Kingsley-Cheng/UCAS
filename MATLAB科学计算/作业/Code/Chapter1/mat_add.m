% function mat_add

function A = mat_add(varargin)
A=0;
for i=1:length(varargin)
    A = A + varargin{i};
end
end