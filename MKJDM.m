function [Zs,Zt,w_opt] = MKJDM(X_src,Y_src,X_tar,Y_tar,index_reweight,A,options)
% Reference: Mingsheng Long. Transfer Joing Matching for visual domain adaptation. CVPR 2014.

% Inputs:
%%% X_src          :     source feature matrix, ns * n_feature
%%% Y_src          :     source label vector, ns * 1
%%% X_tar          :     target feature matrix, nt * n_feature
%%% Y_tar          :     target label vector, nt * 1
%%% options        :     option struct
%%%%% lambda       :     regularization parameter
%%%%% dim          :     dimension after adaptation, dim <= n_feature
%%%%% kernel_tpye  :     kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :     bandwidth for rbf kernel, can be missed for other kernels
%%%%% A            :     adaptation matrix from TCA

% Outputs:
%%% acc            :     final accuracy using knn, float
%%% acc_list       :     list of all accuracies during iterations
%%% w_opt          :     linear combination weights for different kernels
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%% Set options
	lambda = options.lambda;              %% lambda for the regularization
	dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
	gamma = options.gamma;                %% gamma is the bandwidth of rbf kernel
	T = 1;                        %% iteration number
    mu=options.mu;   
  
	fprintf('TJM: dim=%d  lambda=%f\n',dim,lambda);

	% Set predefined variables
	X = [X_src',X_tar'];
% 	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	ns = size(X_src,1);
	nt = size(X_tar,1);
	n = ns+nt;
    sigmas=0:0.025:1;
    do_xval=0;
    [~,~,~,w_opt,~,~]=opt_kernel_comb(X_src,X_tar,sigmas,lambda,do_xval);
   

    nst=ns+nt;
    K=zeros(nst,nst);
        for ii=1:length(sigmas)
    	K =K+ w_opt(ii)*kernel_jda(kernel_type,X,[],sigmas(ii));
        end
	% Construct kernel matrix


	% Construct centering matrix
	H = eye(n)-1/(n)*ones(n,n);

	% Construct MMD matrix
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    C = length(unique(Y_src));
	M = e*e' ;
	Y_tar_pseudo =Y_tar;
    Cls = Y_tar_pseudo ;
	% Transfer Joint Matching: JTM
	G = speye(n);
	acc_list = [];

%% select some source samples to reweight
		  G(index_reweight,index_reweight) = diag(sparse(1./(sqrt(sum(A(index_reweight,:).^2,2)+eps))));
        %%% Mc [If want to add conditional distribution]
        N = 0;
        if ~isempty(Cls) && length(Cls)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(n,1);
                e(Y_src==c) = 1 / length(find(Y_src==c));
                e(ns+find(Cls==c)) = -1 / length(find(Cls==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end
        if mu == 1
            mu = 0.999;
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');
        
        
	    [A,~] = eigs(K*M*K'+lambda*G,K*H*K',dim,'SM');
        
        

       
       
       
	    Z = A'*K;
%          Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
	    Zs = Z(:,1:ns)';
	    Zt = Z(:,ns+1:n)';

	end



% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix

function K = kernel_tjm(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D); 

    case 'sam'
            
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);

    otherwise
        error(['Unsupported kernel ' ker])
end
end



function [w_maxmmd,w_maxrat,w_l2,w_opt,w_med,w_xvalc]=opt_kernel_comb(X,Y,sigmas,lambda,do_xval)

lss=length(sigmas);

% kernel selection method: xvalc
% minimise expeced loss for the binary classifier interpretation of the
% MMD. Selects a single kernel
if do_xval
    num_folds=5;
    mmds=zeros(1,lss);
    for i=1:lss
        %fprintf('xvalmaxmmd for log2(sigma)=%f: ', log2(sigmas(i)));
        % different index permutation for each run
        indices=randperm(size(X,1));

        % do x-validation
        fold_mmds=zeros(1,num_folds);
        for k=1:num_folds
            [train, validation]=cross_val_fold(indices, num_folds, k);

            % training and validation points
            train=[X(train,:);Y(train,:)];
            X_val=X(validation,:);
            Y_val=Y(validation,:);

            % evaluate E[f] on the validation points for all points in P and Q
            % first mean computes function f per validation point (which is
            % mean of kernel values with all training points)
            % second mean computes mean of f values for all validation points
            E_f_P=mean(mean(rbf_dot(X_val,train,sigmas(i)),2));
            E_f_Q=mean(mean(rbf_dot(Y_val,train,sigmas(i)),2));

            fold_mmds(k)=E_f_P-E_f_Q;
        end
        mmds(i)=abs(mean(fold_mmds));
        %fprintf('mmd=%f\n', mmds(i));
    end
    w_xvalc=zeros(lss,1);
    [~,idx_maxxvalc]=max(mmds);
    w_xvalc(idx_maxxvalc)=1;
    fprintf('sigma index for xvalc: %d\n', idx_maxxvalc);
else
    w_xvalc=zeros(lss,1);
end



% kernel selection method: med
% select single kernel that corresponds to the median distance in the data
med_sigma_idx=get_median_sigma_idx(X,Y,sigmas);
w_med=zeros(lss,1);
w_med(med_sigma_idx)=1;
fprintf('sigma index for median: %d\n', med_sigma_idx);


% get rid of MATLAB's optimization terminated message from here
options = optimset('Display', 'off');

% some local variables to make code look nicer
m=size(X,1);
m2=ceil(m/2);

% preallocate arrays for mmds, ratios, etc
mmds=zeros(lss,1);
vars=zeros(lss,1);
ratios=zeros(lss,1);
hh=zeros(lss,m2);

% single kernel selection methods are evaluated for all kernel sizes
for i=1:lss
    % compute kernel diagonals
    K_XX = rbf_dot_diag(X(1:m2,:),X(m2+1:m,:),sigmas(i));
    K_YY = rbf_dot_diag(Y(1:m2,:),Y(m2+1:m,:),sigmas(i));
    K_XY = rbf_dot_diag(X(1:m2,:),Y(m2+1:m,:),sigmas(i));
    K_YX = rbf_dot_diag(X(m2+1:m,:),Y(1:m2,:),sigmas(i));
    
    % this corresponds to the h-statistic that the linear time MMD is the
    % average of
    hh(i,:)=K_XX+K_YY-K_XY-K_YX;
    mmds(i)=mean(hh(i,:));
    
    %variance computed using h-entries from linear time statistic
    vars(i)=var(hh(i,:));
    
    % add lambda to ensure numerical stability
    ratios(i)=mmds(i)/(sqrt(vars(i))+lambda);
    
    % always avoid NaN as the screw up comparisons later. The appear due to
    % divisions by zero. This effectively makes the test fail for the
    % kernel that produced the NaN
    ratios(isnan(ratios))=0;
    ratios(isinf(ratios))=0;
end

% kernel selection method: maxmmd
% selects a single kernel that maxismised the MMD statistic
w_maxmmd=zeros(lss,1);
[~,idx_maxmmd]=max(mmds);
w_maxmmd(idx_maxmmd)=1;
fprintf('sigma index for maxmmd: %d\n', idx_maxmmd);


% kernel selection method: maxrat
% selects a single kernel that maxismised ratio of the MMD by its standard
% deviation. This leads to optimal kernel selection
w_maxrat=zeros(lss,1);
[~,idx_maxrat]=max(ratios);
w_maxrat(idx_maxrat)=1;
fprintf('sigma index for maxrat: %d\n', idx_maxrat);


% kernel selection method: L2
% selects a combination of kernels with a l2 norm constraint that maximises
% the MMD of the combination. Corresponds to maxmmd for convex kernel
% combinations. Note that this corresponds to the 'opt' method below with
% an identity matrix in the optimisation.
w_l2=zeros(lss,1);
warning off
if nnz(mmds>0)>0
    w_l2=quadprog(eye(lss),[],[],[],mmds', 1,zeros(lss,1),[],[],options);
else
    w_l2=quadprog(-eye(lss),[],[],[],mmds',-1,zeros(lss,1),[],[],options);
end
% normalise and apply a low cut to avoid unnecessary computations later
w_l2=w_l2/sum(w_l2);
w_l2(w_l2<1e-7)=0;
[~,max_l2]=max(w_l2);
fprintf('sigma index for maximum weight of l2: %d\n', max_l2);
warning on


% kernel selection method: opt
% selects a combination of kernels via the ratio from maxrat. Corresponds
% to optimal kernel weights

% construct Q matrix and add regulariser to avoid numerical problems
Q=cov(hh');
Q=Q+eye(size(Q))*lambda;
warning off
if nnz(mmds>0)>0 % at least one positive entry
    [wa,~,~]=quadprog(Q,[],[],[],mmds', 1,zeros(lss,1),[],[],options);
else
    [wa,~,~]=quadprog(-Q,[],[],[],mmds',-1,zeros(lss,1),[],[],options);
end
warning on
% normalise and apply low cut to avoid unnecessary computations later
w_opt=zeros(lss,1);
w_opt=wa;
w_opt(w_opt<1e-7)=0;
w_opt=w_opt/sum(w_opt);
[~,max_opt]=max(w_opt);
fprintf('sigma index for maximum weight of opt: %d\n', max_opt);
end



function [H]=rbf_dot_diag(X,Y,deg)
n=size(X,1);

% for now assert that samples have same size
assert(n==size(Y,1));
% 
% H=zeros(n,1);
% 
% 
% for i=1:n
%     dist=X(i,:)-Y(i,:);
%     H(i)=exp(-(dist*dist')/2/deg^2);
% end
dists=X-Y;
dists=dists.*dists;
dists=dists';

% use sum over all columns, if there is only one row, prevent matlab from
% producing a single scalar
if size(dists,1)>1
    dists=sum(dists);
end

% precompute denominator
temp=2*deg^2;
    
    
H=exp(-dists/temp)';
end