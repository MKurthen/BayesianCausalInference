function result = cep_template (X,Y,methodpars)
% function result = cep_template (X,Y,methodpars)
%
% Template for wrappers that run a causal discovery method on a pair of data vectors (X,Y).
%
% It returns its decision about the most likely causal direction as result.decision
% and its confidence value as result.conf.
%
% INPUT:   X                Sample from variable X (Nx1 matrix)
%          Y                Sample from variable Y (Nx1 matrix)
%          methodpars       Struct containing the parameters for the method
%                             (The content of this field may differ across methods).
%
% OUTPUT:  result           Struct that should contain the following fields:
%            X                copy of X
%            Y                copy of Y
%            methodpars       copy of methodpars
%            decision         decision about causal direction
%                               1    means X->Y
%                               -1   means Y->X
%                               0    means no preference for X->Y or Y->X
%                               NAN  means error
%            conf             Confidence that the decision is correct.
%                               (The precise meaning of this value depends on the method. 
%                               In general, the higher the confidence, the more certain 
%                               a method is. It should be -Inf in case decision == NAN.)
%            time             Computation time used.
%            opt              Struct containing any other (partial) results.
%                               (The content of this field may differ across methods).
%
% Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
% All rights reserved.  See the file LICENSE for license terms.

  % prepare result struct
  result = struct;
  result.X = X;
  result.Y = Y;
  result.methodpars = methodpars;
  result.opt = struct;

  if size(X,2) ~= 1 || size(Y,2) ~= 1 % for now, all methods only support univariate X and Y
    result.decision = nan;
    result.conf = -Inf;
  else
    % start measuring computation time
    tic;

    % init (set paths)
    curpath = path;
    % global CEP_PATH_...;
    % addpath(CEP_PATH_...);
    % global CEP_PATH_...;
    % addpath(CEP_PATH_...);

    % set default parameters
    % if ~isfield(methodpars,'par')
    %   methodpars.par = 0;
    % end
    % ...

    % perform causal discovery
    %
    % ...

    % take decision
    %
    % decision = ...;

    % calculate confidence (the higher, the more confidence)
    %
    % conf = ...;

    % save results
    result.time = toc;
    result.decision = decision;
    result.conf = conf;
    % save method-specific results in result.opt
    % result.opt.... = ...;

    % restore path
    path(curpath);
  end

return
