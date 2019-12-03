function K = km_kernel(X1,X2,ktype,kpar)
% KM_KERNEL calculates the kernel matrix between two data sets.
% Input:	- X1, X2: data matrices in row format (data as rows)
%			- ktype: string representing kernel type
%			- kpar: vector containing the kernel parameters
% Output:	- K: kernel matrix
% USAGE: K = km_kernel(X1,X2,ktype,kpar)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2012.
%
% This file is part of the Kernel Methods Toolbox for MATLAB.
% https://github.com/steven2358/kmbox

	switch ktype
		case 'gauss'	% Gaussian kernel
			sgm = kpar;	% kernel width
			
			dim1 = size(X1,1);
			dim2 = size(X2,1);
			
			norms1 = sum(X1.^2,2);
			norms2 = sum(X2.^2,2);
			
			mat1 = repmat(norms1,1,dim2);
			mat2 = repmat(norms2',dim1,1);
			
			distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
	        sgm = sgm / mean(mean(distmat)); % added by jing 24/09/2016, median-distance
			K = exp(-distmat/(2*sgm^2));
			
		case 'gauss-diag'	% only diagonal of Gaussian kernel
			sgm = kpar;	% kernel width
			K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
			
		case 'poly'	% polynomial kernel
	% 		p = kpar(1);	% polynome order
	% 		c = kpar(2);	% additive constant
	        p = kpar; % jing
	        c = 1; % jing
			
			K = (X1*X2' + c).^p;
			
		case 'linear' % linear kernel
			K = X1*X2';
        case 'rbf'
            X1=X1';
            X2=X2';
            n1sq = sum(X1.^2,1);
            n1 = size(X1,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X1'*X1;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X1'*X2;
            end
            K = exp(-kpar*D); 

			
		otherwise	% default case
			error ('unknown kernel type')
    end
end