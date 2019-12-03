function [X_src_new,X_tar_new,A] = TCA(X_src,X_tar,options)
% Reference: Sinno Pan et al. Domain Adaptation via Transfer Component Analysis. TNN 2011.

% Inputs: 
%%% X_src          :    source feature matrix, ns * n_feature
%%% X_tar          :    target feature matrix, nt * n_feature
%%% options        :    option struct
%%%%% lambda       :    regularization parameter
%%%%% dim          :    dimensionality after adaptation (dim <= n_feature)
%%%%% kernel_tpye  :    kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :    bandwidth for rbf kernel, can be missed for other kernels

% Outputs: 
%%% X_src_new      :    transformed source feature matrix, ns * dim
%%% X_tar_new      :    transformed target feature matrix, nt * dim
%%% A              :    adaptation matrix, (ns + nt) * (ns + nt)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%% Set options
	lambda = options.lambda;              
	dim = options.dim;                    
	kernel_type = options.kernel_type;    
	gamma = options.gamma;                

	%% Calculate
	X = [X_src',X_tar'];
%   X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';
	M = M / norm(M,'fro');
	H = eye(n)-1/(n)*ones(n,n);
	if strcmp(kernel_type,'primal')
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
		Z = A' * X;
%         Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	else
	    K = TCA_kernel(kernel_type,X,[],gamma);
	    [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
	    Z = A' * K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	end
end
