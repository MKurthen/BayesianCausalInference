function [train, test] = cvfolds(N,cvk)
% function [train, test] = cvfolds(N,cvk)
%
% Constructs train and test indices for cross-validation on permuted data.
%
% INPUT:   N                number of data points
%          cvk              number of cross-validation folds
%
% OUTPUT:  train            cell vector of length cvk; train{k} contains indices of
%                           data points for training for fold k
%          test             cell vector of length cvk; test{k} contains indices of
%                           data points for testing for fold k
%
% Copyright (c) 2011  Joris Mooij <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  perm = myrandperm(N)';
  K = 0;
  test = cell(cvk,1);
  train = cell(cvk,1);
  m = floor(N / cvk) + 1;
  for k = 1:mod(N,cvk)
    test{k} = perm((K + 1):(K + m));
    train{k} = [perm(1 : K); perm(K + (m + 1):N)];
    K = K + m;
  end
  m = m - 1;
  for k = (1 + mod(N,cvk)):cvk
    test{k} = perm((K + 1):(K + m));
    train{k} = [perm(1 : K); perm(K + (m + 1):N)];
    K = K + m;
  end

return
