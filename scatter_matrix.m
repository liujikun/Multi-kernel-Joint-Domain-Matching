function [Sw,Sb] = scatter_matrix(X,Y)
%% Within and between class Scatter matrix
%% Inputs:
%%% X: data matrix, length * dim
%%% Y: label vector, length * 1
% Outputs:
%%% Sw: With-in class matrix, dim * dim
%%% Sb: Between class matrix, dim * dim
    X = X';
	dim = size(X,1);
	class_set = unique(Y);
	C = length(class_set);
	mean_total = mean(X,2);
	Sw = zeros(dim,dim);
	Sb = zeros(dim,dim);
	for i = 1 : C
		Xi = X(:,Y == class_set(i));
		mean_class_i = mean(Xi,2);
		Hi = eye(size(Xi,2)) - 1/(size(Xi,2)) * ones(size(Xi,2),size(Xi,2));
		Sw = Sw + Xi * Hi * Xi';
		Sb = Sb + size(Xi,2) * (mean_class_i - mean_total) * (mean_class_i - mean_total)';
	end
end