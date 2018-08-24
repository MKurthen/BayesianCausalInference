function Xdisc = discretize(X,nvalues)
% function Xdisc = discretize(X,nvalues)
%
% Discretizes onedimensional data X by repetiviely merging the 
% values for which the sum of the absolute error caused by the
% merge is minimal
%
% Input:  X        Nx1 data matrix (N data points, dimensionality 1)
%         nvalues  number of different values remaining after
%                  discretization
%
% Output: Xdisc   discretized version of X
%
% Copyright (c) 2014  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  assert(size(X,2) == 1);

  lut = unique(sort(X));
  assert(nvalues <= length(lut));
  nlut = zeros(size(lut));
  for i=1:length(lut)
    nlut(i) = length(find(X == lut(i)));
  end

  while length(lut) > nvalues
    N = length(lut);
    cost = zeros(length(lut) - 1,1);
    for i=1:length(lut)-1
      newlut = (nlut(i)*lut(i) + nlut(i+1)*lut(i+1)) / (nlut(i) + nlut(i+1));
      cost(i) = nlut(i) * abs(lut(i) - newlut) + nlut(i+1) * abs(lut(i+1) - newlut);
    end
%    [lut nlut]
%    cost
    imin = argmin(cost);
    lut = [lut(1:imin-1); (nlut(imin)*lut(imin) + nlut(imin+1)*lut(imin+1)) / (nlut(imin) + nlut(imin+1)); lut(imin+2:N)];
    nlut = [nlut(1:imin-1); nlut(imin)+nlut(imin+1); nlut(imin+2:N)];
  end

  Xdisc = X;
  for i=1:length(X)
    Xdisc(i) = lut(argmin(abs(X(i) - lut)));
  end

return
