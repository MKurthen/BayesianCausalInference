function H = entropy(X,method)
% function H = entropy(X,method)
%
% Estimates the differential Shannon entropy of a sample
%
% INPUT:
%   X:        Nx1 vector of values
%   method:   Different methods are supported:
%               'KSG':  equation (4) in A. Kraskov, H. Stoegbauer, and P. Grassberger: 
%                       "Estimating Mutual Information", PHYSICAL REVIEW E 69, 066138 (2004)
%                       Note: this article contains a sign error
%                       Note: only the unique values of X are used, the duplicates are ignored
%             Wrapped estimators from ITE toolbox (must have been initialized):
%               'Shannon_kNN_k', 'Shannon_Edgeworth', 'Shannon_spacing_V', 'Shannon_spacing_Vb',
%               'Shannon_spacing_Vpconst', 'Shannon_spacing_Vplin', 'Shannon_spacing_Vplin2',
%               'Shannon_spacing_LL', 'Shannon_KDP', 'Shannon_MaxEnt1', 'Shannon_MaxEnt2',
%               'Shannon_PSD_SzegoT', 'Shannon_expF', 'Shannon_spacing_VKDE'
%
% OUTPUT:
%   H:        Estimate of the differential Shannon entropy of X
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  if strcmp(method,'KSG')
    % See (4) in A. Kraskov, H. Stoegbauer, and P. Grassberger (2003):
    [X1,indXs] = sort(X);
    N1 = length(X1);
    H = 0.0;
    for i = 1:N1-1
      dX = X1(i+1) - X1(i);
      if dX ~= 0.0
        H = H + log(abs(dX));
      end
    end
    H = H / (N1 - 1) + psi(N1) - psi(1);
  elseif strcmp(method,'Shannon_kNN_k') || strcmp(method,'Shannon_Edgeworth') || strcmp(method,'Shannon_spacing_V') || strcmp(method,'Shannon_spacing_Vb') || strcmp(method,'Shannon_spacing_Vpconst') || strcmp(method,'Shannon_spacing_Vplin') || strcmp(method,'Shannon_spacing_Vplin2') || strcmp(method,'Shannon_spacing_LL') || strcmp(method,'Shannon_KDP') || strcmp(method,'Shannon_MaxEnt1') || strcmp(method,'Shannon_MaxEnt2') || strcmp(method,'Shannon_PSD_SzegoT') || strcmp(method,'Shannon_expF') || strcmp(method,'Shannon_spacing_VKDE')
    co = feval(['H' method '_initialization'], 1);
    H = feval(['H' method '_estimation'],X',co);
  else
    error('entropy.m: unknown method');
  end

return
