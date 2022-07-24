%% kercov - This function implements the kernelized covariance representation proposed in [1]
%
% INPUT
% x : d-by-T matrix representing the time serie of T acquisitions,
%    being each of them a d-dimensional vector.
% kerspec : structure which encodes the kernel type and (eventually) its
%           parameters
%
%           kerspec.type : it specifies which types of kernel should be used.
%           Can be either a polynomial (kerspec.type = 'pol'), Vovk
%           (kerspec.type = 'Vovk') or an exponential-dot product kernel
%           (kerspec.type = 'exp').
%
%           kerspec.deg : degree of the kernel
%
%           kerspec.bias : bias parameter (only used for the polynomial
%           kernel)
%
%           kerspec.sigma : kernel bandwidth (only used for the
%           exponential-dot product kernel)
%
% reg : regularization parameter to prevent the kernelized covariance matrix
%       to be negative definite. The default value is reg = 10^(-7).
%
% OUTPUT
% C : d-by-d matrix corresponding to the kernelized covariance
%     representation, computed according to the method developed in [1].

% Jacopo Cavazza
% Copyright 2016 Jacopo Cavazza [ jacopo.cavazza-at-iit.it ]
% Please, email me if you have any question.

% [1] Jacopo Cavazza, Andrea Zunino, Marco San Biagio, Vittorio Murino
%     Kernelized Covariance for Action Recognition
%     International Conference on Pattern Recognition (ICPR), 2016

function C = kercov(x,kerspec,reg)

%Setting the regularization parameter
if ~exist('reg','var')
    reg = 10e-7;
end

%Kernel definition and computation
switch(kerspec.type)
    case 'pol'
        K = x.^kerspec.deg + kerspec.bias;
    case 'Vovk'
        if ~isfield(kerspec,'deg') || (isfield(kerspec,'deg') && (isempty(kerspec.deg) || isnan(kerspec.deg)))
            K = 1./(1-x+eps);
        else
            K = (1 - x.^(kerspec.deg))./(1-x+eps);
        end
    case 'exp'
        sigma = mean(sqrt(abs(x(:))));
        if isfield(kerspec,'sigma')
            sigma = sigma*kerspec.sigma;
        end
        K = exp(x/(sigma^2+eps));
end

%Efficient implementation for equation (7) in the paper.
t = size(x,2);
D = sum(K,2);
C = (K*(K') - D*(D')/t)/(t-1);

%Regularizing the kernelized covariance imposing positiveness on the
%eigenvalues
[U,S] = eig(C);
C = U*diag(real(log(diag(S)+reg)))*U';