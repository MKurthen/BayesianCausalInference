function f = igci(x,y,refMeasure,estimator)
% Performs causal inference in a deterministic scenario (see [1] and [2] for details)
% 
% USAGE:
%   f = igci(x,y,refMeasure,estimator)
% 
% INPUT:
%   x          - m x 1 observations of x
%   y          - m x 1 observations of y
%   refMeasure - reference measure to use:
%                  1: uniform
%                  2: Gaussian
%   estimator -  estimator to use:
%                  1: entropy (eq. (12) in [1]),
%                  2: integral approximation (eq. (13) in [1]).
%                  3: new integral approximation (eq. (22) in [2]) that should
%                     deal better with repeated values
% 
% OUTPUT: 
%   f < 0:       the method prefers the causal direction x -> y
%   f > 0:       the method prefers the causal direction y -> x
% 
% EXAMPLE: 
%   x = randn(100,1); y = exp(x); igci(x,y,2,1) < 0
%
%
% Copyright (c) 2010-2015  Povilas Daniušis, Joris Mooij
% All rights reserved.  See the file LICENSE for license terms.
% ----------------------------------------------------------------------------
%
% [1]  P. Daniusis, D. Janzing, J. M. Mooij, J. Zscheischler, B. Steudel,
%      K. Zhang, B. Schoelkopf:  Inferring deterministic causal relations.
%      Proceedings of the 26th Annual Conference on Uncertainty in Artificial 
%      Intelligence (UAI-2010).  
%      http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf
% [2]  J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf
%      Distinguishing cause from effect using observational data: methods and benchmarks
%      arXiv:1412.3773v2, submitted to Journal of Machine Learning Research
%      http://arxiv.org/abs/1412.3773v2


if nargin ~= 4
  help igci
  error('Incorrect number of input arguments');
end

% ignore complex parts
x = real(x);
y = real(y);

% check input arguments
[m, dx] = size(x);
if min(m,dx) ~= 1
  error('Dimensionality of x must be 1');
end
if max(m,dx) < 2
  error('Not enough observations in x (must be >= 2)');
end

[m, dy] = size(y);
if min(m,dy) ~= 1
  error('Dimensionality of y must be 1');
end
if max(m,dy) < 2
  error('Not enough observations in y (must be >= 2)');
end

if length(x) ~= length(y)
	error('Lenghts of x and y must be equal');
end

switch refMeasure
  case 1
    % uniform reference measure
    x = (x - min(x)) / (max(x) - min(x));
    y = (y - min(y)) / (max(y) - min(y));
  case 2
    % Gaussian reference measure
    x = (x - mean(x)) ./ std(x);
    y = (y - mean(y)) ./ std(y);
  otherwise
    warning('Warning: unknown reference measure - no scaling applied');
end       
  
switch estimator
  case 1
    % difference of entropies

    [x1,indXs] = sort(x);
    [y1,indYs] = sort(y);

    n1 = length(x1);
    hx = 0.0;
    for i = 1:n1-1
      delta = x1(i+1)-x1(i);
      if delta
        hx = hx + log(abs(delta));
      end
    end
    hx = hx / (n1 - 1) + psi(n1) - psi(1);

    n2 = length(y1);
    hy = 0.0;
    for i = 1:n2-1
      delta = y1(i+1)-y1(i);
      if delta
        hy = hy + log(abs(delta));
      end
    end
    hy = hy / (n2 - 1) + psi(n2) - psi(1);

    f = hy - hx;
  case 2
    % integral-approximation based estimator
    a = 0;
    b = 0;
    [sx,ind1] = sort(x);
    [sy,ind2] = sort(y);

    for i=1:m-1
      X1 = x(ind1(i));  X2 = x(ind1(i+1));
      Y1 = y(ind1(i));  Y2 = y(ind1(i+1));
      if (X2 ~= X1) && (Y2 ~= Y1)   
        a = a + log(abs((Y2 - Y1) / (X2 - X1)));
      end
      X1 = x(ind2(i));  X2 = x(ind2(i+1));
      Y1 = y(ind2(i));  Y2 = y(ind2(i+1));
      if (Y2 ~= Y1) && (X2 ~= X1)
        b = b + log(abs((X2 - X1) / (Y2 - Y1)));
      end
    end

    f = (a - b) / m;
  case 3
    % integral-approximation based estimator
    % improved handling of values that occur multiple times
    f = (improved_slope_estimator(x,y) - improved_slope_estimator(y,x)) / m;
  otherwise 
    error('Unknown estimator');
end

return

function a = improved_slope_estimator(x,y)
% see eq. (22) in http://arxiv.org/abs/1412.3773v2
  m = length(x);
  [sx,ind] = sort(x);

  Xs = []; Ys = []; Ns = [];
  last_index = 1;
  for i=2:m
    if x(ind(i)) ~= x(ind(last_index))
      Xs = [Xs; x(ind(last_index))];
      Ys = [Ys; y(ind(last_index))];
      Ns = [Ns; i - last_index];
      last_index = i;
    end
  end
  Xs = [Xs; x(ind(last_index))];
  Ys = [Ys; y(ind(last_index))];
  Ns = [Ns; m+1 - last_index];

  m = length(Xs);
  a = 0;
  Z = 0;
  for i=1:m-1
    X1 = Xs(i);  X2 = Xs(i+1);
    Y1 = Ys(i);  Y2 = Ys(i+1);
    if (X2 ~= X1) && (Y2 ~= Y1)   
      a = a + log(abs((Y2 - Y1) / (X2 - X1))) * Ns(i);
      Z = Z + Ns(i);
    end
  end
  a = a / Z;
return
