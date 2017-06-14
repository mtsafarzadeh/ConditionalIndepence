function [p, R, bR, cdf] = testCI(X, Y, Z, N)
%% [p, R, bR] = testCI(X, Y, Z, N, fastCompute)
%  Test conditional independence of (X,Z) given Y
% using N bootstrap replicates
% X,Y,Z are vectors of identical lengths
%
% p: p value
% R: test statistic on original data
% bR: tests statistics obtained from Bootstrap
% cdf contains conditional cdf of X,Z computed with and without the
% cond. indep. assumption.

% grid dimension
n = numel(X) ;
nx = 501 ;
x = linspace(-5,5,nx) ;


% Test statistic on original data
% <latex>
% $R$ measure the discrepancy between the joint cdf of $X$ and $Z$, denoted $F_{XZ}$, and
% the one computed based on the conditional independence assumption, denoted $\tilde F_{XZ}$.
% It is defined as $R = \sqrt n \max(|F_{XZ} - \tilde F_{XZ}|)$.
X = (X-mean(X))/std(X) ;
Y = (Y-mean(Y))/std(Y) ;
Z = (Z-mean(Z))/std(Z) ;

FFxz = condCDF2(X,Y,Z,x) ;
lXx = bsxfun(@lt, reshape(X,[n, 1]), reshape(x,[1, nx])) ;
lZz = bsxfun(@lt, reshape(Z,[n, 1]), reshape(x,[1, nx])) ;
Fxz = double(lXx')*double(lZz) / n;
R = sqrt(n)*max(max(abs(Fxz-FFxz))) ;
cdf.trueFXZ = Fxz ;
cdf.condIndepFXZ = FFxz ;

%% generate bootstrap sample

%% Evaluate conditional variance of X and Z given Y
erx = condstd(X,Y) ;
hyx = (n^(-1/6)) * erx ;
erz = condstd(Z,Y) ;
hyz = (n^(-1/6)) * erz ;


%%Bootstrap validation
bR = zeros(N,1) ;
for k=1:N
    %Bootstrap sample for Y and initialization for X and Z
    JY = randi(n,1,n) ;
    bY = Y(JY) ;
    
    %Compute weights for conditionally independent sampling of X and Z given Y    
    dYy = bsxfun(@minus, reshape(Y,[n, 1]), reshape(bY,[1, n])) ;
    wx =  exp(- .5*(dYy/hyx).^2);
    wz =  exp(- .5*(dYy/hyz).^2);
    uw = rand(2,n) ;

    % generate n samples of X,Z conditionally independent given Y
    % based on the kernel density estimates
    % resampling residuals from regression
    zwx = cumsum(wx,1) ;
    zwx = bsxfun(@rdivide, zwx, zwx(end, :)) ;
    zwz = cumsum(wz,1) ;
    zwz = bsxfun(@rdivide, zwz, zwz(end, :)) ;
    JX = sum(bsxfun(@le, zwx, uw(1,:))) + 1; 
    JZ = sum(bsxfun(@le, zwz, uw(2,:))) + 1 ;
    bX = X(JX) ;
    bZ = Z(JZ) ;

    bX = (bX-mean(bX))/std(bX) ;
    bY = (bY-mean(bY))/std(bY) ;
    bZ = (bZ-mean(bZ))/std(bZ) ;
    FFxz = condCDF2(bX,bY,bZ,x) ;
    lXx = bsxfun(@lt, reshape(bX,[n, 1]), reshape(x,[1, nx])) ;
    lZz = bsxfun(@lt, reshape(bZ,[n, 1]), reshape(x,[1, nx])) ;
    Fxz = double(lXx')*double(lZz) / n;
    bR(k) = sqrt(n)*max(max(abs(Fxz-FFxz))) ;
end
p = mean(R < bR) ;

function F = condCDF2(X, Y, Z, x)
%% Computes joint conditional cdf of X and Z given Y assuming that X and Z 
%  are conditionally independent. Values are returned on the grid x.
n = numel(Y) ;
nx = numel(x) ;
% Compute kernel density estimate for Y
h = std(Y) * n^(-1/5) ;
dY = bsxfun(@minus, reshape(Y,[n, 1]), reshape(Y,[1, n])) ;
Ky = exp(-dY.^2/(2*h.^2))/(h*sqrt(2*pi)) ;

lXx = bsxfun(@lt, reshape(X,[n, 1]), reshape(x,[1, nx])) ;
lZx = bsxfun(@lt, reshape(Z,[n, 1]), reshape(x,[1, nx])) ;
FX = bsxfun(@rdivide, lXx' * Ky, reshape(sum(Ky,1), [1,n])) ;
FZ = bsxfun(@rdivide, lZx' * Ky, reshape(sum(Ky,1), [1,n])) ;
F = FX*FZ'/n ;
