function Xsmooth = undiscretize(X)
% function Xsmooth = undiscretize(X)
%
% "Un"discretizes data X by adding random noise
%
% Input:  X   Nxd data matrix (N data points, dimensionality d)
%
% Output: Xsmooth   undiscretized version of X
%
% Copyright (c) 2011  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  N = size(X,1);
  lut = unique(sort(X));
  dlut = diff(lut);
  dlut(length(lut),1) = dlut(length(lut)-1,1);
  lut = [lut dlut];

  Xsmooth = X;
  for i = 1:N
    j = find(lut >= X(i),1,'first');
    Xsmooth(i) = Xsmooth(i) + rand(1,1) * lut(j,2);
  end
return
