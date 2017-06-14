function [s, xpred, xres] = condstd(X,Y)
%% Evaluate conditional variance of X given Y
% using polynomial regression. Estimate polynom order using validation set
% split data set into training and validation

n = numel(X) ;

J = randperm(n) ;
J1 = J(1:floor(n/2)) ;
J2 = J(floor(n/2)+1:end) ;
e0 = std(X) ;

% estimate best polynomial order
warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
warning('off', 'MATLAB:polyfit:RepeatedPoints');

k0 = 0 ;
kmax = min(15, floor(n/10)) ;
for k=1:kmax
    exc = false ;
    try
    P = polyfit(Y(J1),X(J1),k) ;
    catch
        exc = true ;
    end
    er = sqrt(mean((X(J2)-polyval(P, Y(J2))).^2)) ;
    if ~exc && er < e0
        e0 = er; 
        k0 = k ;
    end
end

% Estimate conditional variance on the complete training data
% using optimal order
P = polyfit(Y,X,k0) ;
xpred = polyval(P,Y) ;
xres = X - xpred ;
s = sqrt(mean((xres).^2)) ;
